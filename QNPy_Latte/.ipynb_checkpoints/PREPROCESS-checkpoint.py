import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
from math import ceil
import os
import glob 
import warnings
import dill
from scipy.signal import medfilt

import numpy as np
import pandas as pd

import os
import pandas as pd
import numpy as np

def calculate_polynomial_fit_and_filter(df,sigma_threshold=0.25):
    'Calculate the 5th order polynomial for the magnitude column and remove points above threshold'
    
    # Fit a 5th order polynomial
    polynomial_coeffs = np.polyfit(df.mjd-df.mjd.iloc[0], df.mag, 5)
    fitted_values = np.polyval(polynomial_coeffs, df.mjd-df.mjd.iloc[0])
    
    # Calculate residuals (errors)
    residuals = df.mag - fitted_values

    # Filter out points with residuals greater than sigma threshold
    filtered_df = df[(np.abs(residuals) <= sigma_threshold)]
    
    #Removed Percentage
    removed_perc = 1- (len(filtered_df)/len(df)) 
    
    return filtered_df, removed_perc

def clean_outliers_median(input_folder, output_folder, threshold = 0.25,median = False):
    """
    Clean outliers by removing extreme outliers, applying median filter and removing points above 3sigma threshold from 
    5th degree polynomial.

    Parameters:
        input_folder (str): Path to the folder containing input CSV files.
        output_folder (str): Path to the folder where cleaned CSV files will be saved.
        threshold (float, optional): Threshold for outlier detection in terms of standard deviations. Default is 0.25.
    """
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through each CSV file in the input folder
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".csv"):
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename)
            dataframe = pd.read_csv(input_filepath)
            
            #Sort by date and reset the indices
            dataframe = dataframe.sort_values(by = 'mjd')
            dataframe.reset_index(inplace = True,drop = True)

            #As per Sanchez-Saez et. al. 2021, drop all rows where the magitude error is greater than 1
            dataframe = dataframe[dataframe.magerr < 1]
            
            #Following methodology of Tachibana and Graham et al. 2020
            window_size = 3
            if median:
                dataframe['mag'] = medfilt(dataframe['mag'],window_size)
                dataframe['magerr'] = medfilt(dataframe['magerr'],window_size)
            
            flag = 0
            while flag == 0:
                filtered_df,removed_perc = calculate_polynomial_fit_and_filter(dataframe,threshold)
                if removed_perc > 0.1:
                    threshold += 0.1
                else:
                    flag = 1
                    
            filtered_df.to_csv(output_filepath,index=False)
                
    print('Light Curves Cleaned..')

def backward_pad_curves(folder_path, output_folder, desired_observations=100,verbose = 0):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the list of CSV files in the input folder
    csv_files = [filename for filename in os.listdir(folder_path) if filename.endswith('.csv')]

    # Initialize variables to store maximum number of rows and average values per curve
    max_rows = 0
    average_mag_dict = {}
    average_magerr_dict = {}
    first_column_header = None

    # Iterate over the CSV files to find the maximum number of rows and calculate averages per curve
    for filename in csv_files:
        file_path = os.path.join(folder_path, filename)
        try:
            # Read the CSV file and load the data into a pandas DataFrame
            data = pd.read_csv(file_path)

            # Get the number of rows in the DataFrame
            num_rows = data.shape[0]

            # Determine the header of the first column
            first_column_header = data.columns[0]

            # Assume the second and third columns are mag and magerr
            # Calculate average mag and magerr per curve
            average_mag_dict[filename] = data.iloc[:, 1].mean()
            average_magerr_dict[filename] = data.iloc[:, 2].mean()

            # Update the maximum row count
            if num_rows > max_rows:
                max_rows = num_rows

        except pd.errors.EmptyDataError:
            print(f"Error: Empty file encountered: {filename}")

    # Create new DataFrames with backward padding and ensure a minimum of desired_observations
    new_data_dict = {}

    # Iterate over the CSV files
    for filename in csv_files:
        try:
            # Read the CSV file and load the data into a pandas DataFrame
            data = pd.read_csv(os.path.join(folder_path, filename))

            # Calculate the number of missing rows to reach desired_observations
            missing_rows = desired_observations - len(data)

            # Check the header of the first column and add the appropriate values
            if first_column_header.lower() == 'mjd':
                data['mjd'] = data['mjd'].astype(float)  # Ensure "mjd" column is numeric
                mjd_increment = 0.2  # Specify the MJD increment here
                last_mjd = data.iloc[-1, 0]
                extra_data = pd.DataFrame({
                    first_column_header: np.arange(last_mjd + mjd_increment, last_mjd + mjd_increment * (missing_rows + 1), mjd_increment),
                    'mag': data.iloc[-1, 1],     # Backward-fill the last available mag value
                    'magerr': data.iloc[-1, 2]   # Backward-fill the last available magerr value
                })
            else:
                last_value = data.iloc[-1, 0]
                extra_data = pd.DataFrame({
                    first_column_header: np.arange(last_value + 1, last_value + 1 + missing_rows),
                    'mag': data.iloc[-1, 1],     # Backward-fill the last available mag value
                    'magerr': data.iloc[-1, 2]   # Backward-fill the last available magerr value
                })

            data = pd.concat([data, extra_data], ignore_index=True)

            # Pad to exactly 100 points if the longest curve has fewer than 100 points
            if max_rows < desired_observations:
                pad_rows = desired_observations - len(data)
                if first_column_header.lower() == 'mjd':
                    mjd_increment = 0.2
                    extra_data = pd.DataFrame({
                        first_column_header: np.arange(data.iloc[-1, 0] + mjd_increment, data.iloc[-1, 0] + mjd_increment * (pad_rows + 1), mjd_increment),
                        'mag': average_mag_dict[filename],
                        'magerr': average_magerr_dict[filename]
                    })
                else:
                    extra_data = pd.DataFrame({
                        first_column_header: np.arange(data.iloc[-1, 0] + 1, data.iloc[-1, 0] + 1 + pad_rows),
                        'mag': average_mag_dict[filename],
                        'magerr': average_magerr_dict[filename]
                    })
                data = pd.concat([data, extra_data], ignore_index=True)

            # Save the new DataFrame to a new CSV file in the output folder with the original filename
            output_file = os.path.join(output_folder, filename)
            data.to_csv(output_file, index=False)
            
            if verbose>0:
                print(f"Created new file: {output_file}")
        except pd.errors.EmptyDataError:
            if verbose>0:
                print(f"Error: Empty file encountered: {filename}")


def transform(data):
    x_data = np.array(data['mjd'])
    y_data = np.array(data['mag'])
    z_data = np.array(data['magerr'])
    
    # If magerr is 0 replace it with 0.01*mag
    for i in range(len(z_data)):
        if z_data[i] == 0:
            z_data[i] = 0.01 * y_data[i]
    
    Mmean = np.mean(y_data)
    
    ### mapping time and flux to interval [-2,2]
    a = -2
    b = 2

    Ax = min(x_data)
    Bx = max(x_data)

    Ay = min(y_data)
    By = max(y_data)
   
    # Make series with added noise
    y_dataeplus = y_data + z_data
    # Make series with subtracted noise
    y_dataeminus = y_data - z_data

    Ayeplus = min(y_dataeplus)
    Byeplus = max(y_dataeplus)

    Ayeminus = min(y_dataeminus)
    Byeminus = max(y_dataeminus)

    x_data = (x_data - Ax) * (b-a) / (Bx - Ax) + a
    y_data = (y_data - Ay) * (b-a) / (By - Ay) + a
    
    y_dataeplus1 = (y_dataeplus - Ayeplus) * (b-a) / (Byeplus - Ayeplus) + a
    y_dataeminus1 = (y_dataeminus - Ayeminus) * (b-a) / (Byeminus - Ayeminus) + a

    data_result = {'time': x_data,
                   'cont': y_data,
                   'conterr': z_data}
    data_result = pd.DataFrame(data_result)

    dataeplus_result = {'time': x_data,
                        'cont': y_dataeplus1,
                        'conterr': z_data}
    dataeplus_result = pd.DataFrame(dataeplus_result)

    dataeminus_result = {'time': x_data,
                         'cont': y_dataeminus1,
                         'conterr': z_data}
    dataeminus_result = pd.DataFrame(dataeminus_result)

    return data_result, dataeplus_result, dataeminus_result, Ax, Bx, Ay, By


import os
import warnings
import dill

def transform_and_save(files, data_src, data_dst, transform,augment = False,trcoeff_file = None):
    number_of_points = []
    counter = 0
    trcoeff = []

    for file in files:
        lcName = file.split(".")[0]
        tmpDataFrame = pd.read_csv(os.path.join(data_src, file))
        
        # Catch runtime warnings in transform function
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                data_result, dataeplus_result, dataeminus_result, Ax, Bx, Ay, By = transform(tmpDataFrame)
            except Warning as e:
                print('warning found:', e)
                print(lcName)
                continue
            except Exception as e:
                print('error found:', e)
                print(lcName)
                continue
        
        if data_result is not None:
            counter += 1
            number_of_points.append(len(data_result.index))
            
            # Save the original data
            if augment:
                filename = os.path.join(data_dst, lcName + '_original.csv')
            else:
                filename = os.path.join(data_dst, lcName + '.csv')
            data_result.to_csv(filename, index=False)
            
            if augment:
                # Save the plus light curves
                filename_plus = os.path.join(data_dst, lcName + '_plus.csv')
                dataeplus_result.to_csv(filename_plus, index=False)

                # Save the minus light curves
                filename_minus = os.path.join(data_dst, lcName + '_minus.csv')
                dataeminus_result.to_csv(filename_minus, index=False)
            
            trcoeff.append([lcName, Ax, Bx, Ay, By])
    
    if trcoeff_file is None:
        trcoeff_file = 'trcoeff.pickle'
    dill.dump(trcoeff, file=open(trcoeff_file, "wb"))
    return number_of_points, trcoeff