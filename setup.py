from setuptools import setup, find_packages

setup(
    name='QNPy_Latte',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
    'astropy>=5.3.4',
    'cycler>=0.11.0',
    'dill>=0.3.4',
    'matplotlib>=3.5.2',
    'MiniSom>=2.3.1',
    'numpy>=1.21.5',
    'pandas>=1.4.4',
    'plotly>=5.9.0',
    'scikit_learn>=1.0.2',
    'scipy>=1.9.1',
    'seaborn>=0.12.2',
    'torch>=1.13.0',
    'tqdm>=4.64.1',
    'tslearn>=0.6.2']
    authors = 'Aman Raju',
    author_email='rajuaman@gmail.com',
    description='Latent ATTEntive Neural Processes for Quasar Light Curves with parametric recovery',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/rajuaman1/QNPy_Latte',  # Project's URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
