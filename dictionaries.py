# data output path
data_out=('../')

# data input path
data_file = '/data/users/hgilmour/olr/olr_1h/olr_merge.nc' #this is the 1 hourly mean olr data for the first 20 days of 1998

# set up the parameters to convert from olr to tb
a = 1.228

# b
b = -1.106e-3

# stefan boltzmann constant
sigma = 5.67e-8 # W m^-2 K^-4

# path for the precipitation data
precip = '/data/users/hgilmour/total_precip/precip_instant/total_precip_1998_test.nc' 

# threshold values to be tested in sensitivity analysis
threshold_values = [225, 230, 235, 240, 241, 245]

# set up the parameters for the dictionaries
position_threshold = 'weighted_diff'
sigma_threshold = 0.5
target = 'minimum'
threshold = [225]
n_min_threshold = 1975
method = 'watershed'
v_max = 60
stubs = 7
order = 1
extrapolate = 0
memory = 0
adaptive_stop = 0.2
adaptive_step = 0.95
subnetwork_size = 15
method_linking = 'predict'
