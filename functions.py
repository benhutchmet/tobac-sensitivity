# Functions for the main program

# Import the relevant modules
import iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import iris.quickplot as qplt
import iris.plot as iplt
import datetime
import shutil
from six.moves import urllib
from pathlib import Path
from math import pi
import trackpy
from iris.time import PartialDateTime
import tobac
import warnings

# First set up the function for constraining OLR
def load_olr_data(data_file):
    """
    Loads in CPM 1 hourly OLR data and sets it up for use.
    
    Parameters:
    data_file (str): The path to the OLR data file.
    
    Returns:
    iris.cube.Cube: The OLR data.
    iris.coords.DimCoord: The time coordinate of the OLR data.
    list: The datetimes of the OLR data.
    iris.Constraint: The constraint for the first time step of the OLR data.
    """
    olr = iris.load_cube(data_file)

    #constraining the dataset by time so it runs quicker:
    week = iris.Constraint(time=lambda cell: cell.point.day <= 7)
    olr = olr.extract(week)
    olr.coord('time').bounds = None #REMOVING BOUNDS FROM TIME TO SEE IF THIS HELPS THE TYPEERROR

    # Remove coord system or else the animations don't run (suggested by AVD team)
    olr.coord('latitude').coord_system = None
    olr.coord('longitude').coord_system = None

    # Code from the AVD team
    time = olr.coord('time')
    datetimes = time.units.num2date(time.points)
    con = iris.Constraint(time=datetimes[0])
    olr.extract(con)


    return olr, time, datetimes, con

# Function to set up the directory to save outputs and plots
def setup_directories():
    """
    Sets up directories to save output and plots.
    """
    savedir = Path("Save")
    if not savedir.is_dir():
        savedir.mkdir()
    plot_dir = Path("Plot")
    if not plot_dir.is_dir():
        plot_dir.mkdir()

    return savedir, plot_dir


# Proposed wrapper function for calculating dxy
# Given the longitude, latitude and olr data, calculate the spatial and temporal resolution of the input data.
def calculate_dxy(longitude, latitude, olr):
    """
    Calculates the spatial and temporal resolution of the input data.
    
    Parameters:
    longitude (numpy.ndarray): The longitudes of the input data.
    latitude (numpy.ndarray): The latitudes of the input data.
    olr (iris.cube.Cube): The input data.
    
    Returns:
    float: The spatial resolution of the input data.
    float: The temporal resolution of the input data.
    """
    R = 6.3781e6
    dx = np.gradient(longitude)[1]
    dx = dx * (pi / 180) * R * np.cos(latitude * pi / 180)
    dy = np.gradient(latitude)[0]
    dy = dy * (pi / 180) * R
    dxy, dt = tobac.get_spacings(olr, grid_spacing=4500, time_spacing=3600)
    
    return dxy, dt

# Function for calculating the brightness temperatures from the OLR data
def calculate_temperatures(a, b, sigma, olr):
    """
    Calculates the temperatures of the input data.
    
    Parameters:
    a (float): The value of a.
    b (float): The value of b.
    sigma (float): The value of sigma.
    olr (iris.cube.Cube): The input data.
    
    Returns:
    iris.cube.Cube: The temperatures of the input data.
    iris.cube.Cube: The tb_var of the input data.
    iris.cube.Cube: The tb of the input data.
    """
    tf = (olr.data/sigma)**(0.25)
    tb_var = (-a + np.sqrt(a**2 + 4*b*tf.data))/(2*b)
    
    tb = olr.copy()
    tb.data = tb_var.data

    return tf, tb_var, tb

# Function for loading the precip data
# Very similar to loading olr data
# But doesn't modify precip.coord system
def load_precip_data(precip_path):
    """
    Loads in precipitation data and sets it up for use.
    
    Parameters:
    precip_path (str): The path to the precipitation data file.
    
    Returns:
    iris.cube.Cube: The precipitation data.
    iris.coords.DimCoord: The time coordinate of the precipitation data.
    list: The datetimes of the precipitation data.
    iris.Constraint: The constraint for the first time step of the precipitation data.
    """
    precip = iris.load_cube(precip_path)
    week = iris.Constraint(time=lambda cell: cell.point.day <= 7)
    precip = precip.extract(week)
    precip.coord('time').bounds = None
    time = precip.coord('time')
    datetimes = time.units.num2date(time.points)
    con = iris.Constraint(time=datetimes[0])
    precip.extract(con)
    return precip, time, datetimes, con

# Function for setting up parameters_features
def setup_parameters_features(position_threshold, sigma_threshold, target, threshold, n_min_threshold):
    """
    Sets up the parameters for feature detection.
    
    Parameters:
    position_threshold (str): The position threshold.
    sigma_threshold (float): The sigma threshold.
    target (str): The target.
    threshold (list): The threshold.
    n_min_threshold (int): The minimum number of threshold.
    
    Returns:
    dict: The parameters for feature detection.
    """
    parameters_features = {}
    parameters_features['position_threshold'] = position_threshold
    parameters_features['sigma_threshold'] = sigma_threshold
    parameters_features['target'] = target
    parameters_features['threshold'] = threshold
    parameters_features['n_min_threshold'] = n_min_threshold
    return parameters_features

# Function for setting up parameters segmentation
def setup_parameters_segmentation(target, method, threshold):
    """
    Sets up the parameters for segmentation.
    
    Parameters:
    target (str): The target.
    method (str): The method.
    threshold (int): The threshold.
    
    Returns:
    dict: The parameters for segmentation.
    """
    parameters_segmentation = {}
    parameters_segmentation['target'] = target
    parameters_segmentation['method'] = method
    parameters_segmentation['threshold'] = threshold
    return parameters_segmentation

# Function for setting up parameters linking
def setup_parameters_linking(v_max, stubs, order, extrapolate, memory, adaptive_stop, adaptive_step, subnetwork_size, method_linking):
    """
    Sets up the parameters for linking.
    
    Parameters:
    v_max (int): The maximum velocity.
    stubs (int): The minimum number of timesteps for a tracked cell to be reported.
    order (int): The order.
    extrapolate (int): The extrapolation.
    memory (int): The memory.
    adaptive_stop (float): The adaptive stop.
    adaptive_step (float): The adaptive step.
    subnetwork_size (int): The subnetwork size.
    method_linking (str): The method for linking.
    
    Returns:
    dict: The parameters for linking.
    """
    parameters_linking = {}
    parameters_linking['v_max'] = v_max
    parameters_linking['stubs'] = stubs
    parameters_linking['order'] = order
    parameters_linking['extrapolate'] = extrapolate
    parameters_linking['memory'] = memory
    parameters_linking['adaptive_stop'] = adaptive_stop
    parameters_linking['adaptive_step'] = adaptive_step
    parameters_linking['subnetwork_size'] = subnetwork_size
    parameters_linking['method_linking'] = method_linking
    return parameters_linking

# Function which performs the feature detection
def perform_feature_detection(tb, dxy, savedir, parameters_features):
    """
    Performs feature detection on input data and saves results to file.
    
    Parameters:
    tb (iris.cube.Cube): The input data.
    dxy (tuple): The spatial resolution of the input data.
    savedir (str): The directory to save the output files.
    parameters_features (dict): The parameters for feature detection.
    
    Returns:
    tobac.utils.FeatureDetection: The features detected in the input data.
    """
    savedir = Path(savedir)
    if not savedir.is_dir():
        savedir.mkdir()
    
    # Feature detection and save results to file:
    print('starting feature detection')
    Features = tobac.feature_detection_multithreshold(tb, dxy, **parameters_features)
    Features.to_hdf(savedir / 'Features.h5', 'table')
    print('feature detection performed and saved')
    
    return Features

# Function which performs the segmentation
def perform_segmentation(tb, dxy, savedir, parameters_segmentation, Features):
    """
    Performs segmentation on input data and saves results to file.
    
    Parameters:
    tb (iris.cube.Cube): The input data.
    dxy (tuple): The spatial resolution of the input data.
    savedir (str): The directory to save the output files.
    parameters_segmentation (dict): The parameters for segmentation.
    Features (tobac.utils.FeatureDetection): The features detected in the input data.
    
    Returns:
    tuple: The mask and features segmented from the input data.
    """
    
    # Perform segmentation and save results to files:
    Mask_tb, Features_tb = tobac.segmentation_2D(Features, tb, dxy, **parameters_segmentation)
    print('segmentation tb performed, start saving results to files')
    iris.save([Mask_tb], savedir / 'Mask_Segmentation_tb.nc', zlib=True, complevel=4)
    Features_tb.to_hdf(savedir / 'Features_tb.h5', 'table')
    print('segmentation tb performed and saved')
    
    return Mask_tb, Features_tb

# Function which performs the linking
# and saves the results to file
def perform_linking(Features, tb, dt, dxy, savedir, parameters_linking):
    """
    Performs linking on input data and saves results to file.
    
    Parameters:
    Features (tobac.utils.FeatureDetection): The features detected in the input data.
    tb (iris.cube.Cube): The input data.
    dt (float): The time resolution of the input data.
    dxy (tuple): The spatial resolution of the input data.
    savedir (str): The directory to save the output files.
    parameters_linking (dict): The parameters for linking.
    
    Returns:
    pandas.DataFrame: The tracks detected in the input data.
    """

    # Perform linking and save results to file:
    Track = tobac.linking_trackpy(Features, tb, dt=dt, dxy=dxy, **parameters_linking)
    Track["longitude"] = Track["longitude"] - 360
    Track.to_hdf(savedir / 'Track.h5', 'table')
    print('linking performed and saved')
    
    return Track

# Function which performs the analysis
def perform_analysis(Features, Mask_tb, Track, parameters_features):
    """
    Performs analysis on input data and returns results as a dictionary.
    
    Parameters:
    Features (pandas.DataFrame): The features detected in the input data.
    Mask_tb (iris.cube.Cube): The mask segmented from the input data.
    Track (pandas.DataFrame): The tracks detected in the input data.
    parameters_features (dict): The parameters for feature detection.
    
    Returns:
    dict: A dictionary containing the results of the analysis.
    """
    results = {}
    
    # Number of features detected:
    CC_Features = Features[Features['threshold_value'] < (parameters_features['threshold'].value + 1)]
    results['num_features'] = CC_Features.count()[0]
    
    # Average size of segmented areas associated with feature:
    mean_gridcells = Features['ncells'].mean()
    mean_area = mean_gridcells * (4.5 ** 2)
    results['mean_area_own_eqn'] = mean_area
    
    area = tobac.analysis.calculate_area(Features, Mask_tb, method_area='latlon')
    mean_area = area.mean()
    results['mean_area_tobac_analysis'] = mean_area
    
    # Max size of segmented areas associated with feature:
    max_gridcells = Features['ncells'].max()
    max_area = max_gridcells * (4.5 ** 2)
    results['max_area'] = max_area
    
    # Min size of segmented areas associated with feature:
    min_gridcells = Features['ncells'].min()
    min_area = min_gridcells * (4.5 ** 2)
    results['min_area'] = min_area
    
    # Number of tracks detected:
    results['num_tracks'] = len(Track['cell'].dropna().unique()) - 1
    
    # Average lifetime of tracks:
    lifetime = (tobac.analysis.lifetime_histogram(Track, bin_edges=np.arange(1, 2000, 200), density=False, return_values=True)[3])
    lifetime_hrs = lifetime / 60
    lifetime_mean = lifetime_hrs.mean()
    results['mean_lifetime'] = lifetime_mean
    
    # Max lifetime of tracks:
    lifetime_max = lifetime_hrs.max()
    results['max_lifetime'] = lifetime_max
    
    # Min lifetime of tracks:
    lifetime_min = lifetime_hrs.min()
    results['min_lifetime'] = lifetime_min
    
    # Mean velocity of MCSs:
    velocity = tobac.analysis.calculate_velocity(Track, method_distance='latlon')
    velocity['v'].replace([np.inf, -np.inf], np.nan, inplace=True)
    velocity['v'].dropna(inplace=True)
    results['mean_velocity'] = velocity['v'].mean()
    
    # Max velocity of MCSs:
    results['max_velocity'] = velocity['v'].max()
    
    # Min velocity of MCSs:
    results['min_velocity'] = velocity['v'].min()
    
    return results

# Function which performs the sensitivity analysis
def perform_sensitivity_analysis(tb, savedir, parameters_features, parameters_segmentation, parameters_linking, threshold_values):
    """
    Performs sensitivity analysis for different values of parameters_features['threshold'] and parameters_segmentation['threshold'].
    
    Parameters:
    tb (iris.cube.Cube): The input data.
    savedir (str): The directory to save the output files.
    parameters_features (dict): The parameters for feature detection.
    parameters_segmentation (dict): The parameters for segmentation.
    parameters_linking (dict): The parameters for linking.
    threshold_values (list): A list of threshold values to use for sensitivity analysis.
    
    Returns:
    pandas.DataFrame: A DataFrame containing the results of the analysis for each value of 'threshold'.
    """
    savedir = Path(savedir)
    if not savedir.is_dir():
        savedir.mkdir()
    
    results = []
    for threshold in threshold_values:
        # Set up parameters:
        parameters_features['threshold'] = threshold
        parameters_segmentation['threshold'] = threshold
        
        # Feature detection:
        Features = tobac.feature_detection.precip_features(tb, **parameters_features)
        
        # Segmentation:
        Mask_tb = tobac.segmentation.segmentation_all(Features, tb, **parameters_segmentation)
        
        # Linking:
        Track = tobac.linking_trackpy(Features, tb, dt=parameters_linking['dt'], dxy=parameters_linking['dxy'], **parameters_linking)
        Track["longitude"] = Track["longitude"] - 360
        
        # Analysis:
        analysis_results = perform_analysis(Features, Mask_tb, Track)
        analysis_results['threshold'] = threshold
        results.append(analysis_results)
        
    # Save results to file:
    results_df = pd.DataFrame(results)
    results_df.to_csv(savedir / 'sensitivity_analysis.csv', index=False)
    
    return results_df