import xarray as xr
import numpy as np 
import glob
import copy 
from tqdm import tqdm 

def make_patches(feature_file_path='./',label_file_path='./',n_samples=1,patch_size=32,variable_list=[],grid_size=[1059,1799]):
    """
    This function randomly generates images from some input CONUS wide files
    """
    
    #grab all possible input files 
    feature_files = glob.glob(feature_file_path + '/2016*/*.nc')
    feature_files.sort()
    
    #************
    #grab all possible label files
    #Grab files associated with the files in the feature files list
    label_files = []
    for feature_file in feature_files:
        date = feature_file[-16:-8]
        hr = feature_file[-7:-5]
        label_file = glob.glob(f'label_file_path/%s_%s00.nc' % (date, hour))
        if (label_file == []):
            #remove feature file from feature files
            feature_file.remove(feature_file_path + '/' + date + '/' + date + '_' + hr + '00.nc')
    
    #specify the gridsize    
    halfstep = int(patch_size/2)
    y_max = grid_size[1]-halfstep
    x_max = grid_size[0]-halfstep
    
    #loop over each of the samples we are generating
    for n in tqdm(np.arange(0,n_samples)):
        #grab one random hour 
        file_index = np.random.randint(0,len(feature_files))
        
        #grab the input feature file associated with that hour
        feature_file = feature_files[file_index]
        #open it 
        ds = xr.open_dataset(feature_file)
        
        #**************
        #will need some code to do the same with the output labels 
        label_file = label_files[file_index]
        ds_labels = xr.open_dataset(label_file)
        
        #grab a random x,y start point 
        x_i = np.random.randint(halfstep,x_max)
        y_i = np.random.randint(halfstep,y_max)
        
        #grab that patch 
        patch = ds.isel(x=slice(x_i-halfstep,x_i+halfstep),y=slice(y_i-halfstep,y_i+halfstep))
        labels_patch = ds_labels.isel(x=slice(x_i-halfstep,x_i+halfstep),y=slice(y_i-halfstep,y_i+halfstep))
        mesh_class_patch = labels_patch.mesh
        mesh_90_patch = labels_patch.mesh_90
        
        #need to also include time, lat, lon, relative frequency and non-zero mean size into patch
        #calculte relataive frequency
        relative_frequency = (mesh_90_patch.values.count_nonzero())/(patch_size**2)
        #calculate non-zero mean hail size
        non_zero_mean_size = np.mean(mesh_90_patch.values)
        #calculate latitude
        lat = labels_patch.latitude[0]
        #calculate longitude
        lon = labels_patch.longitude[0]
        #calculate time
        date = feature_file[-16:-8]
        hr = feature_file[-7:-5]
        year = date[-16:-12]
        month = date[-12:-10]
        day = date[-10:-8]
        time = netCDF4.date2num(np.datetime64('%s-%s-%sT%s:00' %(year,month,day,hr)),'seconds since 1970-01-01')
        
        #put all the above values into dataarray
        patch["mesh_90"] = (("x","y","n_samples"), mesh_90_patch)
        patch["mesh_classification"] = (("x","y","n_samples"), mesh_class_patch)
        patch["rel_freq"] = (("n_samples"), relative_frequency)
        patch["mean_size"] = (("n_samples"), non_zero_mean_size)
        patch["lat"] = (("n_samples"), lat)
        patch["lon"] = (("n_samples"), lon)
        patch["time"] = (("n_samples"), time)
        
        
        if n == 0:
            patches = copy.deepcopy(patch.expand_dims(dim='n_samples'))
        else:
            patches = xr.concat([patches,patch.expand_dims(dim='n_samples')],dim='n_samples')
            
    return patches
    
    
    

if(classification == True):
    #for the classification problem:
    patches = make_patches(feature_file_path='ml_input/', label_file_path = 'ml_class_labels/', n_samples=100)
else:
    #for the regression problem:
    patches = make_patches(feature_file_path='ml_input/', label_file_path = 'ml_regress_labels/', n_samples=100)

#save patches to an output file
patches.to_netcdf('ml_patches/patch_test.nc').compute()






