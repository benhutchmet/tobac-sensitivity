#This script is used to test the sensitivity of tobac parameters on the tracking output. Here, 1h olr CPM data is used.

#NEED TO UPDATE TO INCLUDE PRECIP AND COLD CORE FEATURES

#========================================================================

 

#import modules:

#===============

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

import trackpy

from iris.time import PartialDateTime

 

import tobac #tobac package cloned from https://github.com/tobac-project/tobac.git

 

import warnings

warnings.filterwarnings('ignore', category=UserWarning, append=True)

warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)

warnings.filterwarnings('ignore', category=FutureWarning, append=True)

warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

 

 

#load in CPM 1 hrly olr data + set it up ready to use:

#==============================================

data_out=('../')

data_file = '/data/users/hgilmour/olr/olr_1h/olr_merge.nc' #this is the 1 hourly mean olr data for the first 20 days of 1998

print(data_file)

 

olr=iris.load_cube(data_file)

 

 

#constraining the dataset by time so it runs quicker:

week=iris.Constraint(time=lambda cell: cell.point.day<=7) #just the first 7 days

olr = olr.extract(week)

 

olr.coord('time').bounds=None #REMOVING BOUNDS FROM TIME TO SEE IF THIS HELPS THE TYPEERROR

# Remove coord system or else the animations don't run (suggested by AVD team)

olr.coord('latitude').coord_system = None

olr.coord('longitude').coord_system = None

 

#code that AVD team sent:

time = olr.coord('time')

datetimes = time.units.num2date(time.points)

con = iris.Constraint(time=datetimes[0])

olr.extract(con)

 

 

print(olr)

 

#Set up directory to save output and plots:

savedir=Path("Save")

if not savedir.is_dir():

    savedir.mkdir()

plot_dir=Path("Plot")

if not plot_dir.is_dir():

    plot_dir.mkdir()     

 

 

#calculating dxy

from math import pi

longitude,latitude=np.meshgrid(olr.coord('longitude').points,olr.coord('latitude').points)

R=6.3781e6

dx=np.gradient(longitude)[1]

dx=dx*(pi/180)*R*np.cos(latitude*pi/180)

dy=np.gradient(latitude)[0]

dy=dy*(pi/180)*R

print(dx)

print(dy)

 

 

# Determine temporal and spatial sampling of the input data:

dxy,dt=tobac.get_spacings(olr,grid_spacing=4500,time_spacing=3600) #time spacing = 1 hour

 
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

 

#Convert from olr to Tb using Ohring and Gruber empirical formula:

# # (1984) as given in Yang and Slingo (2001)

    # Tf = tb(a+b*Tb) where a = 1.228 and b = -1.106e-3 K^-1

    # OLR = sigma*Tf^4

    # where sigma = Stefan-Boltzmann constant = 5.67x10^-8 W m^-2 K^-4

a = 1.228

b = -1.106e-3

sigma = 5.67e-8 # W m^-2 K^-4

 

tf = (olr.data/5.67e-8)**(0.25)

tb_var = (-1.228 + np.sqrt(1.228**2 + 4*-1.106e-3*tf.data))/(2*-1.106e-3)

 

tb=olr.copy()

tb.data=tb_var.data

print(tb)

 

 

#load in CPM 1 hrly precip data + set it up ready to use:

#==============================================

data_out=('../')

precip = '/data/users/hgilmour/total_precip/precip_instant/total_precip_1998_test.nc' #this is the 3 hourly mean olr data for the first 20 days of 1998

print(precip)

 

precip=iris.load_cube(precip)

 

#constraining the dataset by time so it runs quicker:

week=iris.Constraint(time=lambda cell: cell.point.day<=7) #just the first 7 days

precip = precip.extract(week)

 

precip.coord('time').bounds=None #REMOVING BOUNDS FROM TIME TO SEE IF THIS HELPS THE TYPEERROR

# Remove coord system or else the animations don't run (suggested by AVD team)

#precip.coord('latitude').coord_system = None

#precip.coord('longitude').coord_system = None

 

#code that AVD team sent:

time = precip.coord('time')

datetimes = time.units.num2date(time.points)

con = iris.Constraint(time=datetimes[0])

precip.extract(con)

 

 

print(precip)

 

 

# Determine temporal and spatial sampling of the input data:

#===========================================================

#calculating dxy

from math import pi

longitude,latitude=np.meshgrid(tb.coord('longitude').points,tb.coord('latitude').points)

R=6.3781e6

dx=np.gradient(longitude)[1]

dx=dx*(pi/180)*R*np.cos(latitude*pi/180)

dy=np.gradient(latitude)[0]

dy=dy*(pi/180)*R

print(dx)

print(dy)

 

dxy,dt=tobac.get_spacings(tb,grid_spacing=4500,time_spacing=3600) #time spacing = 1 hour

 

# TOBAC:

#==================

#Parameter features:

parameters_features={}

parameters_features['position_threshold']='weighted_diff'

parameters_features['sigma_threshold']=0.5

parameters_features['target']='minimum'

parameters_features['threshold']=[225] #olr threshold equivalent to Tb=225K based on stefan boltzmann equation (145 for 225K, 91 for 200K, 74 for 190K)

parameters_features['n_min_threshold']=1975 # number of grid points for 40,000km^2 area (7792m = 1 grid space. 4500m x 4500m = 20250000m^2. 40,000km^2 = 4x10^10m^2. 4x10^10 / 20250000 = 1975 (1975 grid cells per 40,000km^2 area)

 

# Feature detection and save results to file:

print('starting feature detection')

Features=tobac.feature_detection_multithreshold(tb,dxy,**parameters_features)

Features.to_hdf(savedir / 'Features.h5','table')

print('feature detection performed and saved')

 

 

# Segmentation:

parameters_segmentation={}

parameters_segmentation['target']='minimum'

parameters_segmentation['method']='watershed'

parameters_segmentation['threshold']=225

 

 

# Perform segmentation and save results to files:

Mask_tb,Features_tb=tobac.segmentation_2D(Features,tb,dxy,**parameters_segmentation)

print('segmentation tb performed, start saving results to files')

iris.save([Mask_tb], savedir / 'Mask_Segmentation_tb.nc', zlib=True, complevel=4)

Features_tb.to_hdf(savedir / 'Features_tb.h5', 'table')

print('segmentation tb performed and saved')

 

 

# Linking:

parameters_linking={}

parameters_linking['v_max']=60 #(velocity of 60 m s-1 is referenced in https://journals.ametsoc.org/view/journals/mwre/126/6/1520-0493_1998_126_1630_lcvomc_2.0.co_2.xml#i1520-0493-126-6-1630-f01 study)

parameters_linking['stubs']=7 #minimum number of timesteps for a tracked cell to be reported (equivalent to 6 hours)

parameters_linking['order']=1

parameters_linking['extrapolate']=0

parameters_linking['memory']=0

parameters_linking['adaptive_stop']=0.2

parameters_linking['adaptive_step']=0.95

parameters_linking['subnetwork_size']=15

parameters_linking['method_linking']= 'predict'

 

 

 

# Perform linking and save results to file:

Track=tobac.linking_trackpy(Features,tb,dt=dt,dxy=dxy,**parameters_linking)

Track["longitude"]=Track["longitude"]-360

Track.to_hdf(savedir / 'Track.h5','table')


# Proposed function for performing linking
# and saving results to file:
def linking_trackpy_wrapper(Features, tb, dt, dxy, v_max, stubs, order, extrapolate, memory, adaptive_stop, adaptive_step, subnetwork_size, method_linking):
    """
    A wrapper function for tobac.linking_trackpy that performs linking and returns a pandas DataFrame with the resulting tracks.
    
    Parameters:
    Features (pandas DataFrame): A DataFrame containing the features to be linked.
    tb (float): The time interval between frames.
    dt (float): The maximum distance between features in consecutive frames.
    dxy (float): The maximum distance between features in the same frame.
    v_max (float): The maximum velocity of a feature.
    stubs (int): The number of consecutive frames a feature must be present in to be considered a track.
    order (int): The order of the polynomial used for fitting the feature positions.
    extrapolate (bool): Whether to extrapolate the feature positions.
    memory (int): The number of frames to remember when linking.
    adaptive_stop (float): The adaptive stopping criterion for linking.
    adaptive_step (float): The adaptive step size for linking.
    subnetwork_size (int): The size of the subnetwork used for linking.
    method_linking (str): The method used for linking.
    
    Returns:
    pandas DataFrame: A DataFrame containing the resulting tracks. - Is this correct harri?
    """
    parameters_linking = {
        'v_max': v_max,
        'stubs': stubs,
        'order': order,
        'extrapolate': extrapolate,
        'memory': memory,
        'adaptive_stop': adaptive_stop,
        'adaptive_step': adaptive_step,
        'subnetwork_size': subnetwork_size,
        'method_linking': method_linking
    }
    
    Track = tobac.linking_trackpy(Features, tb, dt=dt, dxy=dxy, **parameters_linking)
    Track["longitude"] = Track["longitude"] - 360
    return Track

 

 

# ANALYSIS:

#===============

#number of features detected:

CC_Features=Features[Features['threshold_value']<226]

print("{}".format(CC_Features.count())+" number of features dectected")

 

#average size of segmented areas associated with feature:

mean_gridcells=Features_tb['ncells'].mean()

#print(mean_gridcells)

mean_area=mean_gridcells*(4.5**2)

print("{}".format(mean_area)+" km^2 (mean area - USING OWN EQN)")

 

area=tobac.analysis.calculate_area(Features,Mask_tb,method_area='latlon')

mean_area=area.mean()

print("{}".format(mean_area)+" km^2 (mean area - USING TOBAC ANALYSIS FUNCTION")

 

 

#max size of segmented areas associated with feature:

max_gridcells=Features_tb['ncells'].max()

#print(max_gridcells)

max_area=max_gridcells*(4.5**2)

print("{}".format(max_area)+" (max area)")

 

#min size of segmented areas associated with feature:

min_gridcells=Features_tb['ncells'].min()

#print(min_gridcells)

min_area=min_gridcells*(4.5**2)

print("{}".format(min_area)+" (min area)")

 

#number of tracks detected:

print("{}".format(len(Track['cell'].dropna().unique())-1)+" tracks")

 

#average lifetime of tracks:

lifetime = (tobac.analysis.lifetime_histogram(Track,bin_edges=np.arange(1,2000,200),density=False,return_values=True)[3])

lifetime_hrs=lifetime/60

lifetime_mean=lifetime_hrs.mean()

print("{}".format(lifetime_mean)+" hours (mean lifetime of MCS)")

 

#max lifetime of tracks:

lifetime_max=lifetime_hrs.max()

print("{}".format(lifetime_max)+" hours (max lifetime of MCS)")

 

#min lifetime of tracks:

lifetime_min=lifetime_hrs.min()

print("{}".format(lifetime_min)+" hours (min lifetime of MCS)")

 

#mean velocity of MCSs:

velocity=tobac.analysis.calculate_velocity(Track,method_distance='latlon')

velocity['v'].replace([np.inf, -np.inf], np.nan, inplace=True)

velocity['v'].dropna(inplace=True)

print("{}".format(velocity['v'].mean())+" m/s (mean velocity)")

 

#max velocity of MCSs:
print("{}".format(velocity['v'].max())+" m/s (max velocity)")


#min velocity of MCSs:
print("{}".format(velocity['v'].min())+" m/s (min velocity)")