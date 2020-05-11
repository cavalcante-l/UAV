'''
# Workflow to process UAV images

Developed for: Laiza Cavalcante de A. S.
Period: Msc. Engineering Agriculture 
        (Geoprocessing + Remote Sensing + Agriculture)
''' 
#%%
import os
import re
import numpy as np
from tqdm import tqdm
import geopandas as gp
import rasterio
from module_func_rast import *
from module_func_rast import Raster_resample

#%%
''' 
###########################################################
# 1st Step:  
        Clipping UAV rasters by shapefile
'''
# Raster and shp directories
rast_dir = "C:/Users/liz/Pictures/Saved Pictures/"
shp_dir = "C:/Users/liz/Pictures/Saved Pictures/Cultivar_BRS1003IPRO_31982_rec10.shp"

# Creating a list of containing files in a given directory
list_rasters = [ os.path.join(rast_dir, raster) for raster in os.listdir(rast_dir) if raster.endswith('.tif') ]

# Iterating over and clipping
for rast in tqdm(list_rasters):
    rast_class = Raster_operations(rast, shp_dir)

    # Clipping
    new_rast = rast_class.clip_raster_by_shp()

#%%
''' 
###########################################################
# 2nd Step:  
        Normalize UAV rasters 
'''
# Ignore division by 0 - avoid errors
np.seterr(divide='ignore', invalid='ignore')

# Directories for rasters, shapes and output
rast_dir = 'D:/Processamento_python/Clipped/trigo/'
out_directory = 'D:/Processamento_python/Normalized/trigo/'

# Creating a list of containing files in a given directory
list_rasters = [ os.path.join(rast_dir, raster) for raster in os.listdir(rast_dir) if raster.endswith('.tif') ]

# Iterating over and clipping
for rast in tqdm(list_rasters):
    rast_class = UAV_funcs(rast)

    # Normalizing
    new_rast = rast_class.band_normalized_t1()


# %%
''' 
###########################################################
# 3rd Step:  
        Vegetation Index from UAV normalized rasters 
'''

def vegetation_index(rastB, rastG, rastR, iv_type):
    '''
    ### Visible Vegetation Index creation
        Vegetation Indexes: Excess Green (EXG), Green Leaf Index (GLI),
        Green Red Veg. Index (GRVI), Modified GRVI (MGRVI),
        Red Green Blue Veg. Index (RGBVI), Visible Atmospherically Resistant Index (VARI)
        Input: Blue, Green, Red rasters and each iv must be created
    '''

    def correct_vi(array_):
        array_[array_<-1] = -1
        array_[array_>1] = 1
        return array_

    with rasterio.Env(num_threads='all_cpus'):
        with rasterio.open(rastB, num_threads='all_cpus') as b:
            with rasterio.open(rastG, num_threads='all_cpus') as g:
                with rasterio.open(rastR, num_threads='all_cpus') as r:
                    info = b.meta.copy()
                    info.update({'nodata': np.nan, 'count': 1, 'compress': 'lzw', 'dtype': 'float32'})

                    # Reading raster by tiles (raster windows)
                    tiles = r.block_windows(1)

                    out_exg = img[2][0][:-4] + '_EXG.tif'
                    out_gli = img[2][0][:-4] + '_GLI.tif'
                    out_grvi = img[2][0][:-4] + '_GRVI.tif'
                    out_mgrvi = img[2][0][:-4] + '_MGBVI.tif'
                    out_rgbvi = img[2][0][:-4] + '_RGBVI.tif'
                    out_vari = img[2][0][:-4] + '_VARI.tif'

                    if 'EXG' == iv_type:
                        print('Creating EXG index')
                        with rasterio.open(out_exg, 'w', **info) as dst:
                            for idx, window in tiles:   
                                b_band = b.read(1, window=window, masked=True)
                                g_band = g.read(1, window=window, masked=True)
                                r_band = r.read(1, window=window, masked=True)         
                            
                                iv_calc = ( (2 * g_band) - r_band - b_band )                          
                                dst.write_band(1, iv_calc, window=window)

                    if 'GLI' == iv_type:
                        print('Creating GLI index')
                        with rasterio.open(out_gli, 'w', **info) as dst:
                                for idx, window in tiles:   
                                    b_band = b.read(1, window=window, masked=True)
                                    g_band = g.read(1, window=window, masked=True)
                                    r_band = r.read(1, window=window, masked=True) 

                                    iv_calc = ( ((2 * g_band) - r_band - b_band) /((2 * g_band) + r_band + b_band) )    
                                    iv_calc = correct_vi(iv_calc)                     
                                    dst.write_band(1, iv_calc, window=window)

                    if 'GRVI' == iv_type:
                        print('Creating GRVI index')
                        with rasterio.open(out_grvi, 'w', **info) as dst:
                            for idx, window in tiles:   
                                b_band = b.read(1, window=window, masked=True)
                                g_band = g.read(1, window=window, masked=True)
                                r_band = r.read(1, window=window, masked=True)         
    
                                iv_calc = ((g_band - r_band)/(g_band + r_band))   
                                iv_calc = correct_vi(iv_calc)                     
                                dst.write_band(1, iv_calc, window=window)

                    if 'MGRVI' == iv_type:
                        print('Creating MGRVI index')
                        with rasterio.open(out_mgrvi, 'w', **info) as dst:
                            for idx, window in tiles:   
                                b_band = b.read(1, window=window, masked=True)
                                g_band = g.read(1, window=window, masked=True)
                                r_band = r.read(1, window=window, masked=True)         
                            
                                iv_calc = ( ((g_band**2) - (r_band**2))/((g_band**2) + (r_band**2)) )                           
                                iv_calc = correct_vi(iv_calc)     
                                dst.write_band(1, iv_calc, window=window)

                    if 'RGBVI' == iv_type:
                        print('Creating RGBVI index')
                        with rasterio.open(out_rgbvi, 'w', **info) as dst:
                            for idx, window in tiles:   
                                b_band = b.read(1, window=window, masked=True)
                                g_band = g.read(1, window=window, masked=True)
                                r_band = r.read(1, window=window, masked=True)         
                            
                                iv_calc = ((g_band**2)-(b_band * r_band))/((g_band**2)+(b_band * r_band))                          
                                iv_calc = correct_vi(iv_calc)     
                                dst.write_band(1, iv_calc, window=window)

                    if 'VARI' == iv_type:
                        print('Creating VARI index')
                        with rasterio.open(out_vari, 'w', **info) as dst:
                            for idx, window in tiles:   
                                b_band = b.read(1, window=window, masked=True)
                                g_band = g.read(1, window=window, masked=True)
                                r_band = r.read(1, window=window, masked=True)         
                            
                                iv_calc = ((g_band - r_band)/(g_band + r_band - b_band))                        
                                iv_calc = correct_vi(iv_calc)     
                                dst.write_band(1, iv_calc, window=window)
    
    return iv_calc

rast_dir = 'C:/Users\liz/Pictures/Saved Pictures/1/'
np.seterr(divide='ignore', invalid='ignore')

lst_rast = [ os.path.join(rast_dir, raster) for raster in os.listdir(rast_dir) if raster.endswith('.tif') ]
dates = set([re.search(r"(\d{8}|\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4}|\d{4}_\d{2}_\d{2})", elm).group() for elm in lst_rast])

rast = []
for date in dates:
    bands, elms = [], []
    for elm in lst_rast:
        if date in elm:
            bands.append(re.search(r"([R|G|B]+)", elm).group())
            elms.append(elm)
    rast.append( (date, bands, elms) )

ivs = ['GRVI', 'GLI', 'VARI', 'RGBVI', 'EXG']

for img in tqdm(rast):
    for iv in tqdm(ivs):
        result = vegetation_index(rastB = img[2][0], 
                        rastG = img[2][1], 
                        rastR = img[2][2],
                        iv_type = iv)

#%%
'''
########################################################### 
# 4th Step:  
        Resample images 
'''

rast_dir = 'C:/Users/liz/Pictures/Saved Pictures/1/'
lst_rast = [ os.path.join(rast_dir, raster) for raster in os.listdir(rast_dir) if raster.endswith('.tif') ]

Sentinel = r'D:\Processamento_python\Sentinel_Level_2A_BOA\soja\20181212_EG_BOA.tif'

# Iterating over and clipping
for rast in tqdm(lst_rast):
    rst_class = Raster_resample(raster_base=Sentinel, 
                                raster_dir=rast, 
                                scale = 132)
    rst_resample = rst_class.resample_by_raster()
    

# %%
