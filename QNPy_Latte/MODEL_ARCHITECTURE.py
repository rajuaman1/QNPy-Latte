import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F
import abc
from .ATTENTION import get_attender

class DeterministicEncoder(nn.Module):
    """The Deterministic Encoder with Attentive Mechanism"""

    def __init__(self,encoding_size,attention = False,attention_type = 'scaledot',activation = 'relu',lstm_size = 0):
        """
        AttnCNP encoder.
        
        Args:
        encoding_size: Dimension of each layer of the MLP
        attention: Whether to use self-attention in the encoder for the context points
        attention_type: The type of attention to use. Choose from 'scaledot' or 'multihead'
        activation: Whether to use ReLu or LeakyReLu activation function
        lstm_size: The size of the encoded LSTM output (0 if no LSTM)
        
        Returns:
        R: MLP (and maybe attention) representation of the input
        """
        super(DeterministicEncoder, self).__init__()
        
        #Linear Layers for MLP
        if lstm_size > 0:
            first_input = lstm_size + 1
        else:
            first_input = 2
        self.linearInput = nn.Linear(first_input, encoding_size)
        self.linear1     = nn.Linear(encoding_size, encoding_size)
        self.linear2     = nn.Linear(encoding_size, encoding_size)
        self.linear3     = nn.Linear(encoding_size, encoding_size)
        self.linear_encoding = nn.Linear(1,encoding_size)

        #ReLu Function
        if activation == 'relu':
            self.relu         = nn.ReLU()
        else:
            self.relu         = nn.LeakyReLU()
        
        self.encoding_size = encoding_size
        self.batchnorm = nn.Identity()
        #Attention Mechanism
        if attention:
            if attention_type == 'scaledot':
                self.self_attender = get_attender(attention_type,encoding_size,encoding_size,encoding_size)
                self.attention_type = 'scaledot'
            else:
                self.self_attender = nn.MultiheadAttention(encoding_size,8,batch_first=True)
                self.attention_type = 'multihead'
        else:
            self.self_attender = None
        
    def forward(self, context_x, context_y):
        """Encodes the inputs into one representation.

        Args:
          context_x, context_y: Tensors of size BATCH_SIZE x NUM_OF_OBSERVATIONS
          
        Returns:
          representation: The encoded representation averaged over all context points.
        """
        # Concatenate x and y along the filter axes
        #context_x = torch.unsqueeze(context_x, -1)
        context_y = torch.unsqueeze(context_y, -1)
        encoder_input = torch.cat((context_x, context_y), dim=-1)

        #Get the dimensions for later use
        batch_size, num_context_points, filter_size = encoder_input.shape
        
        #Reshape to parallelize across observations
        input = torch.reshape(encoder_input, (batch_size * num_context_points, -1))
            
        # Pass through layers
        x = self.relu(self.batchnorm(self.linearInput(input)))
        x = self.relu(self.batchnorm(self.linear1(x)))
        x = self.relu(self.batchnorm(self.linear2(x)))
        x = self.batchnorm(self.linear3(x))
        
        #Reshape back to original size
        rep = torch.reshape(x, (batch_size, num_context_points, self.encoding_size))
        
        #Utilize self attention if indicated
        if self.self_attender is not None:
            k = context_x
            q = context_x
            v = rep
            if self.attention_type == 'scaledot':
                self_attn_rep = self.self_attender(k,q,v)
            else:
                q = self.linear_encoding(q)
                k = self.linear_encoding(k)
                self_attn_rep,_ = self.self_attender(q,k,v)
            rep = self_attn_rep
            
        return rep

class LatentEncoder(nn.Module):
    """The Latent Encoder with capability to make multiple samples"""

    def __init__(self,encoding_size,latent_dim = 8,latent_space_sample = 1,self_attention = False,attention_type = 'scaledot',activation = 'relu',lstm_size = 0,lstm_agg = False):
        """
        LNP encoder.
        
        Args:
        encoding_size: Dimension of each layer of the MLP
        latent_dim: The dimension of the latent space
        latent_space_sample: How many samples to make of the latent space. Recommended to use 20 if not using NPVI
        self_attention: Whether to use self-attention in the encoder for the context points
        attention_type: The type of attention to use. Choose from 'scaledot' or 'multihead'
        activation: Whether to use ReLu or LeakyReLu activation function
        lstm_size: The size of the encoded LSTM output (0 if no LSTM)
        lstm_agg: Whether to use an LSTM to aggregate the 

        Returns:
        None
        """
        super(LatentEncoder, self).__init__()
        #MLP for encoding
        if lstm_size > 0:
            first_input = lstm_size + 1
        else:
            first_input = 2
            
        self.linearInput = nn.Linear(first_input, encoding_size)
        self.linear1     = nn.Linear(encoding_size, encoding_size)
        self.linear2     = nn.Linear(encoding_size, encoding_size)
        self.linear3     = nn.Linear(encoding_size, encoding_size)
        self.linear_encoding = nn.Linear(1,encoding_size)

        if activation == 'relu':
            self.relu         = nn.ReLU()
        else:
            self.relu         = nn.LeakyReLU()
        #Latent Space Encoding
        penultimate_latent_size = int(0.5*(encoding_size+latent_dim))
        self.penultimate_encoder = nn.Linear(encoding_size,penultimate_latent_size)
        #self.mean_and_var_encoder = nn.Linear(penultimate_latent_size,latent_dim*2)
        self.mean_encoder = nn.Linear(penultimate_latent_size,latent_dim)
        self.var_encoder = nn.Linear(penultimate_latent_size,latent_dim)
        self.softplus     = nn.Softplus()
        #Attention
        self.self_attention = self_attention
        if self_attention:
            if attention_type == 'scaledot':
                self.self_attender = get_attender(attention_type,encoding_size,encoding_size,encoding_size)
                self.attention_type = attention_type
            else:
                self.self_attender = nn.MultiheadAttention(encoding_size,8,batch_first=True)
                self.attention_type = attention_type

        else:
            self.self_attender = None
        #LSTM for aggregation
        if lstm_agg is True:
            self.lstm_agg_layer = nn.LSTM(encoding_size,encoding_size,num_layers = 1,batch_first = True)
        self.lstm_agg = lstm_agg
        
        #More Parameters
        self.encoding_size = encoding_size
        self.latent_dim = latent_dim
        self.no_samples = latent_space_sample
        #self.batchnorm = nn.BatchNorm1d(encoding_size)
        self.batchnorm = nn.Identity()
        
    def forward(self, context_x, context_y):
        """Encodes the inputs into one representation.

        Args:
          context_x, context_y: Tensors of size BATCH_SIZE x NUM_OF_OBSERVATIONS

        Returns:
          latent_dist: The underlying latent space distribution
        """
        # Concatenate x and y along the filter axes
        #context_x = torch.unsqueeze(context_x, -1)
        context_y = torch.unsqueeze(context_y, -1)
        encoder_input = torch.cat((context_x, context_y), dim=-1)
        
        #Get the dimensions for later use
        batch_size, num_context_points, filter_size = encoder_input.shape
        
        #Reshape to parallelize across observations
        input = torch.reshape(encoder_input, (batch_size * num_context_points, -1))
        
        # Pass through layers
        x = self.relu(self.batchnorm(self.linearInput(input)))
        x = self.relu(self.batchnorm(self.linear1(x)))
        x = self.relu(self.batchnorm(self.linear2(x)))
        x = self.batchnorm(self.linear3(x))
        
        #Reshape back to original size
        rep = torch.reshape(x, (batch_size, num_context_points, self.encoding_size))
        
        #Utilize self attention if indicated
        if self.self_attention:
            k = context_x
            q = context_x
            v = rep
            if self.attention_type == 'scaledot':
                self_attn_rep = self.self_attender(k,q,v)
            else:
                q = self.linear_encoding(q)
                k = self.linear_encoding(k)
                self_attn_rep,_ = self.self_attender(q,k,v)
            rep = self_attn_rep   
        #Mean pool
        if self.lstm_agg:
            rep,_ = self.lstm_agg_layer(rep)
            rep = rep[:,-1,:]
        else:
            rep = torch.mean(rep, dim=1)
        #Encode into penultimate layer (Halfway between rep and latent space size)
        penultimate_rep = self.relu(self.penultimate_encoder(rep))
        #Get the mean and log var from encoders
        #latent_mean_and_var_comb = self.mean_and_var_encoder(penultimate_rep)
        #latent_mean,latent_log_var = latent_mean_and_var_comb.chunk(chunks=2, dim=-1)
        latent_mean = self.mean_encoder(penultimate_rep)
        latent_log_var = self.var_encoder(penultimate_rep)
        #Bound the variance and exponentiate the variance
        latent_var =  0.01 + 0.99 * self.softplus(latent_log_var)
        #Find the latent variable from the space
        latent_dist = Normal(latent_mean, latent_var)
        return latent_dist    

class DeterministicDecoder(nn.Module):
    """The Decoder."""

    def __init__(self,encoding_size,latent_dim,activation = 'leaky',lstm_size = 0):
        """CNP decoder."""
        super(DeterministicDecoder, self).__init__()
        
        if latent_dim>0 and encoding_size > 0:
            full_encoding_size = latent_dim+encoding_size
        elif latent_dim > 0:
            full_encoding_size = latent_dim
        elif encoding_size > 0:
            full_encoding_size = encoding_size
        if lstm_size > 0:
            input_size = full_encoding_size+lstm_size
        else:
            input_size = full_encoding_size+1
                
        self.linearInput  = nn.Linear(input_size, full_encoding_size)
        self.linear1      = nn.Linear(full_encoding_size, full_encoding_size)
        self.linear2      = nn.Linear(full_encoding_size, full_encoding_size)
        self.linearOutput = nn.Linear(full_encoding_size, 2)
        #self.meanOutput = nn.Linear(full_encoding_size, 1)
        #self.varOutput = nn.Linear(full_encoding_size, 1)
        if activation == 'relu':
            self.relu         = nn.ReLU()
        else:
            self.relu         = nn.LeakyReLU()
        self.softplus     = nn.Softplus()
        self.encoding_size = encoding_size
        #self.batchnorm = nn.BatchNorm1d(full_encoding_size)
        self.batchnorm = nn.Identity()

    def forward(self, representation, target_x):
        """Decodes the individual targets.

        Args:
          representation: The encoded representation of the context
          target_x: The x locations for the target query

        Returns:
          dist: A multivariate Gaussian over the target points.
          mu: The mean of the multivariate Gaussian.
          sigma: The standard deviation of the multivariate Gaussian.
        """
        num_total_points = target_x.shape[1]

        # Concatenate the representation and the target_x
        #target_x = torch.unsqueeze(target_x, -1)
        
        target_x_reshaped = target_x.unsqueeze(0).expand(*representation.shape[:-1], target_x.shape[-1])
        decoder_input = torch.cat((representation, target_x_reshaped), dim=-1)
        no_samples, batch_size, num_context_points, filter_size = decoder_input.shape
        
        input = torch.reshape(decoder_input, (no_samples*batch_size * num_context_points, -1))

        # Pass through layers
        x = self.relu(self.batchnorm(self.linearInput(input)))
        x = self.relu(self.batchnorm(self.linear1(x)))
        x = self.relu(self.batchnorm(self.linear2(x)))
        out = self.linearOutput(x)
        #mu = self.meanOutput(x)
        #log_sigma = self.varOutput(x)

        #Reshape back 
        out = torch.reshape(out, (no_samples,batch_size, num_context_points, 2))
        
        # Get the mean an the variance
        mu, log_sigma = torch.split(out, 1, dim=-1)
        #mu = torch.reshape(mu, (no_samples,batch_size, num_context_points, 1))
        #log_sigma = torch.reshape(log_sigma, (no_samples,batch_size, num_context_points, 1))

        # Bound the variance
        sigma = 0.01 + 0.99 * self.softplus(log_sigma)
        
        # Squeeze last dim
        mu = torch.squeeze(mu, dim=-1)
        sigma = torch.squeeze(sigma, dim=-1)
        
        # Get the distribution
        dist = Normal(mu, sigma)

        return dist, mu, sigma

class FullModel(nn.Module):
    """The AttnLNP model."""

    def __init__(self,encoding_size = 128,latent_dim = 6,latent_mlp_size = 128,cross_attention = True,self_attention = False,
                 attention_type = 'scaledot',latent_space_sample = 1,latent_mode = 'NPVI',lstm_layers = 0,lstm_size = 64,activation = 'leaky',lstm_agg = False,transfer_function_length = 0, parameters_length = 0, classes = 0, replace_lstm_with_gru = False,bidirectional = False):
        """Initializes the model."""
        super(FullModel, self).__init__()
        
        #If both latent and deterministic layer are made, then same encoding size
        if encoding_size is not None and latent_dim is not None:
            latent_mlp_size = encoding_size
        elif latent_dim is None:
            latent_dim = 0
        elif encoding_size is None:
            encoding_size = 0
        #Saving the latent size and encoding size    
        self._latent_size = latent_dim
        self._encoding_size = encoding_size
        self._sample_size = latent_space_sample
        #LSTM Modules
        if lstm_layers > 0:
            if lstm_layers > 1:
                dropout = 0.25
            else:
                dropout = 0
            self._lstm = True
            if replace_lstm_with_gru:
                self._lstm_layer = nn.GRU(1,lstm_size,lstm_layers,batch_first = True,bidirectional = bidirectional,dropout = dropout)
            else:
                self._lstm_layer = nn.LSTM(1,lstm_size,lstm_layers,batch_first = True,dropout = dropout,bidirectional = bidirectional)
            if bidirectional:
                lstm_size = lstm_size*2
            self._no_lstm = lstm_layers
            self._lstm_size = lstm_size
        else:
            self._lstm = False
            self._lstm_size = 0
            lstm_size = 0
        #Initializing the encoders if defined
        if encoding_size > 0:
            self._encoder = DeterministicEncoder(encoding_size,self_attention,attention_type,activation = activation,lstm_size = lstm_size)
            self._cross_attention = cross_attention
            if attention_type == 'scaledot':
                self._cross_attender = get_attender(attention_type,encoding_size,encoding_size,encoding_size)
            else:
                self._cross_attender = nn.MultiheadAttention(encoding_size,8,batch_first = True)
        if latent_dim > 0:
            self._latent = LatentEncoder(latent_mlp_size,latent_dim,\
                                         latent_space_sample,self_attention,attention_type,activation = activation,lstm_size = lstm_size,lstm_agg=lstm_agg)
        self._decoder = DeterministicDecoder(encoding_size,latent_dim,activation = activation,lstm_size = lstm_size)
        #Other Parameters
        self._attention_type = attention_type
        self._latent_samples = latent_space_sample
        self._latent_mode = latent_mode
        if activation == 'relu':
            self._activation = nn.ReLU()
        else:
            self._activation = nn.LeakyReLU()
        self._linear_encoder = nn.Linear(1,encoding_size)
        self._r_z_merger = nn.Linear(encoding_size+latent_dim,latent_dim)
        self._softplus = nn.Softplus()
        #Parametric training
        if transfer_function_length > 0:
            self._tf_head = nn.Sequential(
                nn.Linear(latent_dim+encoding_size,int(latent_dim+transfer_function_length/2)),
                nn.ReLU(),
                nn.Linear(int(latent_dim+transfer_function_length/2),2*transfer_function_length)
            )
            self._use_transfer_function = True
        else:
            self._use_transfer_function = False
        if parameters_length > 0:
            self._parameters_head = nn.Sequential(
                nn.Linear(latent_dim+encoding_size,latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim,2*parameters_length)
            )
            self._use_parameters = True
        else:
            self._use_parameters = False
        if classes > 0:
            self._classifier_head = nn.Sequential(
                nn.Linear(latent_dim,latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim,classes)
            )
            self._use_classifier = True
        else:
            self._use_classifier = False

    def forward(self, context_x, context_y, target_x,target_y = None):
        """Returns the predicted mean and variance at the target points.

        Args:
          context_x: Array of shape BATCH_SIZE x NUM_CONTEXT that contains the x values of the context points.
          context_y: Array of shape BATCH_SIZE x NUM_CONTEXT that contains the y values of the context points.
          target_x: Array of shape BATCH_SIZE x NUM_TARGET that contains the x values of the target points.
          target_y: Array of shape BATCH_SIZE x NUM_TARGET that contains the y values of the target points. (Only provide
          when utilizing NPVI based model)
          
        Returns:
          dist: A multivariate Gaussian over the target points.
          mu: The mean of the multivariate Gaussian.
          sigma: The standard deviation of the multivariate Gaussian.
          kl_loss: The K-L Loss of the latent variable between prior and posterior
          z: The latent vector(s) created from sampling the latent space
        """
        #Apply LSTM on the data if on
        if self._lstm:
            #Apply RNN on context x
            context_x,_ = self._lstm_layer(context_x.unsqueeze(-1))
            #Apply RNN on target x
            target_x,_ = self._lstm_layer(target_x.unsqueeze(-1))
        else:
            context_x = context_x.unsqueeze(-1)
            target_x = target_x.unsqueeze(-1)
            
        #Initializes the latent encoder if it has a finite size
        if self._latent_size != 0:
            dist_prior = self._latent(context_x,context_y)
            #If target_y is provided, then NPVI is activated and sampling of posterior happens
            if target_y is not None:
                dist_post = self._latent(target_x,target_y)
                z = dist_post.rsample([self._sample_size])
                latent_dist = dist_post
                #KL Loss is calculated for optimization
                kl_loss = kl_divergence(dist_post,dist_prior)
                kl_loss = kl_loss.mean(-1)
            else:
                #Else sample from the prior distribution
                z = dist_prior.rsample([self._sample_size])
                kl_loss = None  
                latent_dist = dist_prior
        else:
            #If no latent space, then empty results
            z = None
            kl_loss = None
            latent_dist = None    
        #Initializes the deterministic encoder if it has a finite size
        if self._encoding_size != 0:   
            R = self._encoder(context_x, context_y)
            if self._cross_attention:
                #Use cross attention representation if it exists
                #k = self._activation(self._linear_encoder(context_x.unsqueeze(-1)))
                k = context_x
                q = target_x
                #q = self._activation(self._linear_encoder(target_x.unsqueeze(-1)))
                v = R
                if self._attention_type == 'scaledot':
                    agg_R = self._cross_attender(k,q,v)
                else:
                    q = self._linear_encoder(q)
                    k = self._linear_encoder(k)
                    agg_R,_ = self._cross_attender(q,k,v)
            else:
                agg_R = R.mean(dim = 1)
                agg_R = torch.tile(torch.unsqueeze(agg_R, dim=1),[1, target_x.shape[-2], 1])
        else:
            R = None
            agg_R = None
        
        if z is None:
            #Get the dist, mu and sigma from the decoder
            agg_R = agg_R.unsqueeze(0)
            dist, mu, sigma =  self._decoder(agg_R,target_x)
            agg_R_z = agg_R
        elif R is None:
            z_expand = z.unsqueeze(-2).expand(z.shape[0],z.shape[1],target_x.shape[-2],z.shape[2])
            dist, mu, sigma =  self._decoder(z_expand,target_x)
            agg_R_z = z_expand
        else:
            z_expand = z.unsqueeze(-2).expand(z.shape[0],z.shape[1],target_x.shape[-2],z.shape[2])
            agg_R_expand = agg_R.unsqueeze(0).expand(z.shape[0],agg_R.shape[0],agg_R.shape[1],agg_R.shape[2])
            agg_R_z = torch.cat((agg_R_expand,z_expand),dim = -1)
            #agg_R_z = self._r_z_merger(agg_R_z)
            dist, mu, sigma =  self._decoder(agg_R_z,target_x)
        
        latent_rep_full_curve = agg_R_z.mean(dim=-2)
        predicted_param = None
        predicted_tf = None
        predicted_classes = None
        if self._use_parameters:
            #Get the predicted parameters distribution
            predicted_param_mean_log_var_comb = self._parameters_head(latent_rep_full_curve)
            #Split the mean and sigma
            mu_param, log_sigma_param = predicted_param_mean_log_var_comb.mean(dim = 0).chunk(chunks=2, dim=-1)
            #Bound the variance 
            sigma_param = 0.01 + 0.99 * self._softplus(log_sigma_param)
            # Squeeze last dim
            mu_param = torch.squeeze(mu_param, dim=-1)
            sigma_param = torch.squeeze(sigma_param, dim=-1)
            # Get the distribution
            predicted_param = Normal(mu_param, sigma_param)
        if self._use_transfer_function:
            #Get the predicted transfer function
            predicted_tf_mean_log_var_comb = self._tf_head(latent_rep_full_curve)
            #Split the mean and sigma
            mu_tf, log_sigma_tf = predicted_tf_mean_log_var_comb.mean(dim = 0).chunk(chunks=2, dim=-1)
            # Bound the variance
            sigma_tf = 0.01 + 0.99 * self._softplus(log_sigma_tf)
            # Squeeze last dim
            mu_tf = torch.squeeze(mu_tf, dim=-1)
            #Predicted TF Dist
            predicted_tf = Normal(mu_tf, sigma_tf)
        if self._use_classifier:
            predicted_classes = self._classifier_head(latent_rep_full_curve)
        return dist, mu, sigma,kl_loss, z, R, latent_dist, agg_R_z, predicted_param, predicted_tf, predicted_classes