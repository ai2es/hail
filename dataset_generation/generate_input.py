import xarray as xr
import numpy
import time
import shutil
import os
import glob
import sys

#time the input data generation
tic = time.perf_counter()
#find all the days that we have data for
all_hrrr_dirs = glob.glob('/ourdisk/hpc/ai2es/hail/hrrr/hrrr/*/')
all_hrrr_dirs.sort()

#Index primer indicates the day of data that we are looking at in this particuar run of this code
index_primer = int(sys.argv[1])
#find which day corresponds to index_primer
this_hrrr_dir = all_hrrr_dirs[index_primer]


date = this_hrrr_dir[-9:-1]

year = date[:4]
month = date[4:6]
day = date[6:]

filenames = glob.glob(f"../../ourdisk/hpc/ai2es/hail/hrrr/hrrr/%s/%s_hrrr.t*z.wrfprsf00.grib2" %(date,date))
filenames.sort()

#for each hour:
for orig_filename in filenames:
    
    hour = orig_filename[-19:-17]
    filename = "scratch/tmp_hrrr/%s_hrrr.t%sz.wrfprsf00.grib2" % (date,hour)
    shutil.copyfile(orig_filename, filename)
    
    #open the DataArrays for each surface which has variables that we want
    ds_isobaricInhPa = xr.open_dataset(filename, filter_by_keys={'typeOfLevel': 'isobaricInhPa'})
    ds_atmosphere = xr.open_dataset(filename, filter_by_keys={'typeOfLevel': 'atmosphere'})
    ds_surface_accum = xr.open_dataset(filename, filter_by_keys={'stepType': 'accum', 'typeOfLevel': 'surface'})
    ds_surface_instant = xr.open_dataset(filename, filter_by_keys={'stepType': 'instant', 'typeOfLevel': 'surface'})



    #open each variable of interest

    #isobaricInhPa - surface
    temperature_s = ds_isobaricInhPa.t.sel(isobaricInhPa=1013).values
    dewpoint_temperature_s = ds_isobaricInhPa.dpt.sel(isobaricInhPa=1013).values
    vertical_velocity_s = ds_isobaricInhPa.w.sel(isobaricInhPa=1013).values
    u_component_winds_s = ds_isobaricInhPa.u.sel(isobaricInhPa=1013).values
    v_component_winds_s = ds_isobaricInhPa.v.sel(isobaricInhPa=1013).values
    graupel_s = ds_isobaricInhPa.grle.sel(isobaricInhPa=1013).values

    #isobaricInhPa - 500mb
    temperature_500 = ds_isobaricInhPa.t.sel(isobaricInhPa=500).values
    dewpoint_temperature_500 = ds_isobaricInhPa.dpt.sel(isobaricInhPa=500).values
    vertical_velocity_500 = ds_isobaricInhPa.w.sel(isobaricInhPa=500).values
    u_component_winds_500 = ds_isobaricInhPa.u.sel(isobaricInhPa=500).values
    v_component_winds_500 = ds_isobaricInhPa.v.sel(isobaricInhPa=500).values
    graupel_500 = ds_isobaricInhPa.grle.sel(isobaricInhPa=500).values

    #atmosphere
    reflectivity = ds_atmosphere.refc.values
    vertically_integrated_liquid = ds_atmosphere.veril.values
    if 'hail' in ds_atmosphere:
        hail = ds_atmosphere.hail.values
        lightning = ds_atmosphere.ltng.values

    #surface -> instant
    cape = ds_surface_instant.cape.values
    cin = ds_surface_instant.cin.values

    #surface -> accum
    total_precipitation = ds_surface_accum.tp.values #***zero***
    
    #coordinates
    lats = ds_atmosphere.latitude.values
    lons = ds_atmosphere.longitude.values

    if 'hail' in ds_atmosphere:
        # Create the new DataArray with all variables together
        ds_return = xr.Dataset(
             data_vars=dict(temp_sfc = (["x", "y"], temperature_s),
                           dewtemp_sfc = (["x", "y"], dewpoint_temperature_s),
                           w_sfc = (["x", "y"], vertical_velocity_s),
                           u_sfc = (["x", "y"], u_component_winds_s),
                           v_sfc = (["x", "y"], v_component_winds_s),
                           graup_sfc = (["x", "y"], graupel_s),
                           temp_500 = (["x", "y"], temperature_500),
                           dewtemp_500 = (["x", "y"], dewpoint_temperature_500),
                           w_500 = (["x", "y"], vertical_velocity_500),
                           u_500 = (["x", "y"], u_component_winds_500),
                           v_500 = (["x", "y"], v_component_winds_500),
                           graup_500 = (["x", "y"], graupel_500),
                           refc = (["x", "y"], reflectivity),
                           vertintliq = (["x", "y"], vertically_integrated_liquid),
                           hail = (["x", "y"], hail),
                           ltng = (["x", "y"], lightning),
                           cape = (["x", "y"], cape),
                           cin = (["x", "y"], cin),
                           totalprecip = (["x", "y"], total_precipitation)),
             coords=dict(longitude = (["x", "y"], lons), latitude = (["x", "y"], lats)))
    else:
        # Create the new DataArray with all variables together
        ds_return = xr.Dataset(
             data_vars=dict(temp_sfc = (["x", "y"], temperature_s),
                           dewtemp_sfc = (["x", "y"], dewpoint_temperature_s),
                           w_sfc = (["x", "y"], vertical_velocity_s),
                           u_sfc = (["x", "y"], u_component_winds_s),
                           v_sfc = (["x", "y"], v_component_winds_s),
                           graup_sfc = (["x", "y"], graupel_s),
                           temp_500 = (["x", "y"], temperature_500),
                           dewtemp_500 = (["x", "y"], dewpoint_temperature_500),
                           w_500 = (["x", "y"], vertical_velocity_500),
                           u_500 = (["x", "y"], u_component_winds_500),
                           v_500 = (["x", "y"], v_component_winds_500),
                           graup_500 = (["x", "y"], graupel_500),
                           refc = (["x", "y"], reflectivity),
                           vertintliq = (["x", "y"], vertically_integrated_liquid),
                           cape = (["x", "y"], cape),
                           cin = (["x", "y"], cin),
                           totalprecip = (["x", "y"], total_precipitation)),
             coords=dict(longitude = (["x", "y"], lons), latitude = (["x", "y"], lats)))

    
    if not os.path.exists(f"../../ourdisk/hpc/ai2es/hail/ml_input/%s/" % (date)):
        os.mkdir(os.path.join("../../ourdisk/hpc/ai2es/hail/ml_input", date))
    ds_return.to_netcdf(f"../../ourdisk/hpc/ai2es/hail/ml_input/%s/%s%s%s_%s00.nc" % (date,year,month,day,hour))
    os.remove(f"scratch/tmp_hrrr/%s_hrrr.t%sz.wrfprsf00.grib2" % (date,hour))
    os.remove(f"scratch/tmp_hrrr/%s_hrrr.t%sz.wrfprsf00.grib2.923a8.idx" % (date,hour))
        
toc = time.perf_counter()
print(f"Total time = {toc - tic:0.4f} seconds")