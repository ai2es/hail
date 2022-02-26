import glob 
import numpy as np 
import pandas as pd 
import wget 
import zipfile
import os
import sys
import shutil
from pathlib import Path
from tqdm import tqdm 

def grab_file(date):
     """ This is the main download method. the save path is hard coded for now,
     but could easily be changed to an input param. This will download one specific 
     date and hour from the IEM archive of MRMS. """
     
     save_path = '/ourdisk/hpc/ai2es/hail/tmp_mrms/IEM/'
     killflag = False
     #build URL. Example: https://mrms.agron.iastate.edu/2021/01/04/2021010400.zip
     url_start = 'https://mrms.agron.iastate.edu/'
     url = url_start + date.strftime('%Y') + '/' 
     url += date.strftime('%m')+ '/' 
     url += date.strftime('%d')+ '/'
     url += date.strftime('%Y%m%d%H')+ '.zip'
     
     #check to make sure the directory exists 
     Path(save_path+date.strftime('%Y%m%d')).mkdir(parents=True, exist_ok=True)
     
     filename = save_path+date.strftime('%Y%m%d')+'/' + url[-14:]
     #check to see if file already is downloaded (save some time if it is)
     if os.path.exists(filename):
          blah=1 #placeholder 
     else:
          #add in some exception behavior in case data is missing
          try:
               wget.download(url,out = filename,bar=False)
          except:
               killflag=True
               print('WARNING: URL NOT FOUND {}'.format(url))
               print('or it could be a wget error'.format(url))
               return killflag
     #unzip only the CONUS file (there is Alaska, GUAM and others 
     archive = zipfile.ZipFile(filename)
     for file in archive.namelist():
          if file.startswith(date.strftime('%Y%m%d%H')+'/CONUS/'):
               archive.extract(file, save_path+date.strftime('%Y%m%d'))
     #remove the raw zip file to save space
     os.remove(filename)
     return killflag

#here is the driver python code 
save_path = '/ourdisk/hpc/ai2es/hail/tmp_mrms/IEM/'
#total number of days needed from the IEM Archive
date_range_master = pd.date_range(start='1/4/2021', end='7/1/2021',freq='d')

#takes index from python call to loop over all days 
index_primer = int(sys.argv[1])
this_date=date_range_master[index_primer]
#create the individual hourly instances
date_range = pd.date_range(start=this_date, end=this_date+pd.DateOffset(1),freq='H')
#drop the next day 
date_range = date_range[:-1]

#loop over all 24 hours 
for date in tqdm(date_range[0:1]):
     #download zip file 
     killflag = grab_file(date)
     if killflag:
          continue
     else:
          #need to move things around. No need for two subdirs here. 
          source_dirs = glob.glob(save_path+date.strftime('%Y%m%d')+'/'+date.strftime('%Y%m%d%H')+'/CONUS/*')
          for source_dir in source_dirs:
               #check to make sure the directory exists 
               Path(save_path+date.strftime('%Y%m%d')+source_dir[62:] + '/').mkdir(parents=True, exist_ok=True)
               sources = glob.glob(save_path+date.strftime('%Y%m%d')+'/'+date.strftime('%Y%m%d%H')+'/CONUS/' + source_dir[63:] + '/*')
               for source in sources:
                    grb_file = source[len(source_dir)-len(source)+1:]
                    destination = save_path+date.strftime('%Y%m%d')+source_dir[62:] + '/' + grb_file
                    dest = shutil.move(source, destination)

#remove all empty hourly # dirs
old_dirs = glob.glob(save_path+date.strftime('%Y%m%d') + '/2021*')
for this_dir in old_dirs:
     shutil.rmtree(this_dir,ignore_errors=True)
