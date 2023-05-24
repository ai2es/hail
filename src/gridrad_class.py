import time 
import gc
import numpy as np
import xarray as xr

class gridrad:
    """
    Name: GRIDRAD Python Class
    
    Purpose: This module contains three functions for dealing with Gridded NEXRAD WSR-88D Radar (GridRad) produced by Cameron Homeyer. 
    The original files are stored in a sparse format and can be downloaded from here: https://rda.ucar.edu/datasets/ds841.6/
    This class handles those sparse data, has some QC routines and then provides the user with an xarray dataset.
    
    Primary Author: Randy Chase (@dopplerchase) 
    
    The code (and thresholds) was originally adapted from Cameron Homeyer's code on GridRad.org (http://gridrad.org/software.html) 
    
    Last updated Jan 2023
    
    """
    
    def __init__(self,filename=None,filter=False,toxr=False,timer=False):
        """
        initialize class
        
        ======
        inputs
        ======
        filename: str; path to file of interest
        filter: bool; turn on to turn on QC 
        toxr: bool; return xr dataset
        timer: bool; turns on the timing print statements. 
        
        """
        #store inputs 
        self.filename=filename 
        self.filter_flag=0
        self.ds = None
        self.timer = timer 
        
        #if you provide a filename, go ahead and read the sparse file 
        if self.filename is not None:
            #auto read the file 
            stime = time.time()
            self.read_raw()
            if self.timer:
                print('Time to read file: {}'.format(time.time()-stime))
              
        #if filter flag is on, do QC
        if filter:
            stime = time.time()
            self.filter_sparse()
            if self.timer:
                print('Time to filter: {}'.format(time.time()-stime))
            stime = time.time()
            self.remove_clutter_part1()
            if self.timer:
                print('Time to filter2: {}'.format(time.time()-stime))
            stime = time.time()
            self.Z_H = self.remove_clutter_part2()
            if self.timer:
                print('Time to filter3: {}'.format(time.time()-stime))
                
        #if xarray flag is on, run that 
        if toxr:
            stime = time.time()
            self.build_dataset()
            if self.timer:
                print('Time to build ds: {}'.format(time.time()-stime))

    def read_raw(self):
        
        """ 
        Load the sparse GridRad file. Sparse means only the observed data points are saved. So data are in 1d-vector
        """
        ds = xr.open_dataset(self.filename)
        ds = ds.load()
        ds = ds.rename({'index':'idx'})
        ds = ds.assign_coords({'idx':ds.idx})
        
        self.ds = ds 
        
    def filter_sparse(self):
        
        """ This takes over the filter function in the original code. This does it in the sparse space, which speeds it up by 10x on my machine"""
        
        nz = self.ds.Altitude.shape[0]
        ny = self.ds.Latitude.shape[0]
        nx = self.ds.Longitude.shape[0]
        
        # Extract year from GridRad analysis time string
        year = int((self.ds.attrs['Analysis_time'])[0:4])

        wthresh     = 1.5         # Set default bin weight threshold for filtering by year (dimensionless)
        freq_thresh = 0.6         # Set echo frequency threshold (dimensionless)
        Z_H_thresh  = 15.0        # Reflectivity threshold (dBZ)
        nobs_thresh = 2           # Number of observations threshold

        # Extract dimension sizes
        nx = self.ds.Longitude.shape[0]
        ny = self.ds.Latitude.shape[0]
        nz = self.ds.Altitude.shape[0]

        idx_spar = self.ds.idx.values
        
        ipos= np.where(self.ds.Nradobs.values.ravel()[idx_spar]>0)[0]
        npos = len(ipos)

        if (npos > 0):
            echo_frequency = self.ds.Nradecho.values.ravel()[idx_spar][ipos]/self.ds.Nradobs.values.ravel()[idx_spar][ipos]		# Compute echo frequency (number of scans with echo out of total number of scans)

        inan = np.where(np.isnan((self.ds.Reflectivity.values)))				# Find bins with NaNs 
        nnan = len(inan[0])														# Count number of bins with NaNs

        cond1 = (self.ds.wReflectivity.values  < wthresh)
        cond2 = (self.ds.Reflectivity.values  < Z_H_thresh)
        cond3 = (echo_frequency< freq_thresh)
        cond4 = (self.ds.Nradobs.values.ravel()[idx_spar][ipos] > nobs_thresh)
        ifilter = np.where(~((cond1 & cond2) | (cond3 & cond4)))[0]

        self.ds = self.ds.isel({'Index':ifilter})

    def remove_clutter_part1(self):
        """ This method does some clearning using the dual-pol data and removes some speckles """
        
        # Set fractional areal coverage threshold for speckle identification
        areal_coverage_thresh = 0.32

        self.nz = self.ds.Altitude.shape[0]
        self.ny = self.ds.Latitude.shape[0]
        self.nx = self.ds.Longitude.shape[0]

        # Copy altitude array to 3 dimensions
        zzz = np.tile(self.ds.Altitude.values[:,np.newaxis],(1,self.ny))
        zzz = np.tile(zzz[...,np.newaxis],(1,1,self.nx)).ravel()
        zzz_sparse = zzz[self.ds.idx.values]

        # Light pass at a correlation coefficient decluttering approach first
        if self.ds.DifferentialReflectivity.shape[0] > 0:
            cond1 = (self.ds.Reflectivity.values < 40.0)
            cond2 = (self.ds.CorrelationCoefficient.values < 0.9)
            cond3 = (self.ds.Reflectivity.values < 25.0)
            cond4 = (self.ds.CorrelationCoefficient.values < 0.95)
            cond5 = (zzz_sparse >= 10)
            igood = np.where(~((cond1 & cond2) | ((cond3 & (cond4 & cond5)))))[0]
            self.ds = self.ds.isel({'Index':igood})
            zzz_sparse = zzz_sparse[igood]

        # First pass at removing speckles
        fin = np.zeros((self.nz,self.ny,self.nx),dtype=bool)
        fin1d = np.isfinite(self.ds.Reflectivity.values)
        idx1d = self.ds.idx.values[fin1d]
        z,y,x = np.unravel_index(idx1d,(self.nz,self.ny,self.nx))
        fin[z,y,x] = True

        # Compute fraction of neighboring points with echo
        cover = np.zeros((self.nz,self.ny,self.nx))
        for i in range(-2,3):
            for j in range(-2,3):
                cover += np.roll(np.roll(fin, i, axis=2), j, axis=1)

        cover = cover/25.0

        z,y,x = np.unravel_index(self.ds.idx.values,(self.nz,self.ny,self.nx))

        igood = np.where(cover[z,y,x].ravel() > areal_coverage_thresh)[0]

        self.ds = self.ds.isel({'Index':igood})
    
    def remove_clutter_part2(self,skip_weak_ll_echo = 0):
        """ This method does some more cleaning (ground clutter + below anvil + another speckle pass) """
        # Attempts to mitigate ground clutter and biological scatterers
        if (skip_weak_ll_echo == 0):
            #build ZH array 
            Z_H = self.undo_sparse('Reflectivity')
            
            #build altitude array 
            # Copy altitude array to 3 dimensions
            zzz = np.tile(self.ds.Altitude.values[:,np.newaxis],(1,self.ny))
            zzz = np.tile(zzz[...,np.newaxis],(1,1,self.nx))
            
            # First check for weak, low-level echo
            inan = np.where(np.isnan(Z_H))				# Find bins with NaNs 
            nnan = len(inan[0])															# Count number of bins with NaNs

            if (nnan > 0): 
                (Z_H)[inan] = 0.0

            # Find weak low-level echo and remove (set to NaN)
            ibad = np.where(((Z_H < 10.0) & (zzz <= 4.0)))
            nbad = len(ibad[0])
            if (nbad > 0): 
                (Z_H)[ibad] = float('nan')
                            
            # Replace NaNs that were removed
            if (nnan > 0):
                (Z_H)[inan] = float('nan')
            
            #good data points
            Z_H_1d = Z_H.ravel()[self.ds.idx.values] 
            igood = np.where(~np.isnan(Z_H_1d))[0]
        
            self.ds = self.ds.isel({'Index':igood})

            # Second check for weak, low-level echo
            inan = np.where(np.isnan(Z_H))				# Find bins with NaNs 
            nnan = len(inan[0])															# Count number of bins with NaNs

            if (nnan > 0): (Z_H)[inan] = 0.0

            refl_max   = np.nanmax(Z_H,             axis=0)
            echo0_max  = np.nanmax((Z_H >  0.0)*zzz, axis=0)
            echo0_min  = np.nanmin((Z_H >  0.0)*zzz, axis=0)
            echo5_max  = np.nanmax((Z_H >  5.0)*zzz, axis=0)
            echo15_max = np.nanmax((Z_H > 15.0)*zzz, axis=0)

            # Replace NaNs that were removed
            if (nnan > 0): (Z_H)[inan] = float('nan')

            # Find weak and/or shallow echo
            cond1 = (refl_max   <  20.0)
            cond2 = (echo0_max  <= 4.0)
            cond3 = (echo0_min  <= 3.0)
            cond4 = (refl_max   <  10.0)
            cond5 = (echo0_max  <= 5.0)
            cond6 = (echo0_min  <= 3.0)
            cond7 = (echo5_max  <=  5.0)
            cond8 = (echo5_max  >  0.0)
            cond9 = (echo15_max <= 3.0)
            cond10 = (echo15_max <   2.0)
            cond11 = (echo15_max >  0.0)
            
            ibad = np.where((cond1 & cond2 & cond3) | \
                    (cond4 & cond5 & cond6) | \
                    (cond7 & cond8 & cond9)  | \
                    (cond10 & cond11))
            
            nbad = len(ibad[0])
            if (nbad > 0):
                kbad = (np.zeros((nbad))).astype(int)
                for k in range(0,self.nz):
                    (Z_H)[(k+kbad),ibad[0],ibad[1]] = float('nan')

            #good data points
            Z_H_1d = Z_H.ravel()[self.ds.idx.values] 
            igood = np.where(~np.isnan(Z_H_1d))[0]
        
            self.ds = self.ds.isel({'Index':igood})
                    
        # Find clutter below convective anvils
        k4km = ((np.where(self.ds.Altitude.values >= 4.0))[0])[0]
        fin  = np.isfinite(Z_H)
        cond1 = (fin[k4km,:,:] == 0)
        cond2 = (np.sum(fin[k4km:(self.nz-1),:,:], axis=0) >  0)
        cond3 = (np.sum(fin[0:(k4km-1),:,:], axis=0) >  0)
        
        ibad = np.where((cond1 & cond2 & cond3))
        nbad = len(ibad[0])
        if (nbad > 0):
            kbad = (np.zeros((nbad))).astype(int)
            for k in range(0,k4km+1):
                (Z_H)[(k+kbad),ibad[0],ibad[1]] = float('nan')
                
        #good data points
        Z_H_1d = Z_H.ravel()[self.ds.idx.values] 
        igood = np.where(~np.isnan(Z_H_1d))[0]

        self.ds = self.ds.isel({'Index':igood})

        # Second pass at removing speckles
        fin = np.isfinite(Z_H)

        # Compute fraction of neighboring points with echo
        cover = np.zeros((self.nz,self.ny,self.nx))
        for i in range(-2,3):
            for j in range(-2,3):
                cover += np.roll(np.roll(fin, i, axis=2), j, axis=1)
        cover = cover/25.0

        # Set fractional areal coverage threshold for speckle identification
        areal_coverage_thresh = 0.32
        # Find bins with low nearby areal echo coverage (i.e., speckles) and remove (set to NaN).
        ibad = np.where(cover <= areal_coverage_thresh)
    
        nbad = len(ibad[0])
        if (nbad > 0): 
            (Z_H)[ibad] = float('nan')
            
            
        #good data points
        Z_H_1d = Z_H.ravel()[self.ds.idx.values] 
        igood = np.where(~np.isnan(Z_H_1d))[0]

        self.ds = self.ds.isel({'Index':igood})
        
        # free up some RAM 
        del zzz,ibad,cover,fin,k4km
        gc.collect()
        
    def build_dataset(self):
        """ This is the method that now converts things into a gridded xarray dataset"""
        
        keys_in = ['Reflectivity','SpectrumWidth','AzShear','Divergence','DifferentialReflectivity','DifferentialPhase','CorrelationCoefficient']
        keys_out = ['ZH','SW','VOR','DIV','ZDR','KDP','RHV',]
        
        #build dtime
        #need to convert time
        from pandas import to_datetime 
        from netCDF4 import date2num 
        time = date2num(to_datetime(self.ds.attrs['Analysis_time']),
                          'seconds since 2001-01-01 00:00:00')
        
        
        for i,key in enumerate(keys_in):
            tmp = self.undo_sparse(key)
            
            da = xr.DataArray(tmp[np.newaxis,:,:,:].astype(np.float32), 
                  dims=['time','Altitude','Latitude','Longitude'],
                  coords={'time': [time],
                          'Longitude': self.ds.Longitude.values,
                          'Latitude': self.ds.Latitude.values,
                          'Altitude': self.ds.Altitude.values})
            
            if i == 0:
                self.ds_out = da.to_dataset(name = keys_out[i])
            else:
                self.ds_out[keys_out[i]] = da
            
    def undo_sparse(self,key=None):
        """ this helps build the 3d arrays"""
        nz = self.ds.Altitude.shape[0]
        ny = self.ds.Latitude.shape[0]
        nx = self.ds.Longitude.shape[0]
        z,y,x = np.unravel_index(self.ds.idx.values,[nz,ny,nx])
        Z = np.empty([nz,ny,nx])
        Z[:] = np.nan
        Z[z,y,x] = self.ds[key].values
        return Z