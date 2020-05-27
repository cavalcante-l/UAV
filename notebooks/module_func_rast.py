
'''
DESCRIPTION
  Objective: Funcoes para Recortar um tif, Normalizar, Padronizar e Transformar um Raster pra um tif de pontos. 
  Requisites: Diretorio com rasters e 1 shapefile (para recorte)
  
  Developer: Laiza Cavalcante de Albuquerque Silva
  Period: Msc. Engineering Agriculture (2018-2020)
            (Geoprocessing + Remote Sensing + Agriculture)

'''

###############################################
# Importing packages
###############################################

import os
import fiona
import scipy
import fiona
import rasterio
import numpy as np
import pandas as pd
import geopandas as gp
import rasterio.mask
from matplotlib import pyplot as plt
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.geometry import MultiPolygon, Point
import shapely as sly
from tqdm import tqdm

#incluir a funcao raster to shp (sem ser por pontos)

class Raster_operations():
    '''
    ### Common tasks to deal with rasters:
    Input: raster, shapefile
    Additional criteria: write file or not
    '''
    def __init__(self, img_directory, shp_directory_crop='', writing=True):
        self.__imgdir = img_directory
        self.__shpdir = shp_directory_crop
        self.__write = writing

    def clip_raster_by_shp(self):
        '''
        ### Clip a raster
            Inport: raster path and shapefile path
            It will clip the raster using shp bondaries.
            Please, check if all of them has the same extent
        '''
        # Loading the shapefile
        with fiona.open(self.__shpdir, 'r') as shp:
            print('Shape info \n', shp.schema)
            features = [feature['geometry'] for feature in shp]
 
        # Loading raster
        with rasterio.open(self.__imgdir, 'r+') as tif:
            tif.nodata = np.nan
            #  Cropping it using a rasterio mask
            out_image, out_transform = rasterio.mask.mask(tif, features, crop=True, nodata=np.nan)
            
            # Updating profile information
            out_meta = tif.meta.copy()
            print(out_meta)
            # print('Before changes \n', out_meta)
            out_meta.update({'driver': 'GTiff',
                            'height': out_image.shape[1],
                            'width': out_image.shape[2],
                            'transform': out_transform,
                            'compress': 'lzw'
                            })   

            # Creating new file to save clipped tif
            if self.__write == True:
                output = self.__imgdir[:-4] + '_C.tif'
                with rasterio.open(output, 'w', **out_meta) as dest:
                    dest.write(out_image) 
        
        return out_image

    def raster_nan_corret(self):
        '''
        ### Nodata handling
            Correcting abscence of nodata in raster
            Input: a raster 
            Ex: If nodata is -3.999e-10 will be replaced by np.nan
        '''
        with rasterio.open(self.__imgdir) as tif:
            image = tif.read(1) 
            profile = tif.meta.copy()
            profile.update({'nodata': np.nan})

            # Check if exist nodata
            if np.isnan(np.sum(image)) == True:
                pass
            else:
                # Using 1st value as nodata value
                wrong_nodata = image[0][0]
                # Replacing it by np.nan
                image[np.where( image==wrong_nodata)] = np.nan
            
        # Saving
        if self.__write == True:
            output = self.__imgdir[:-4] + '_Cor.tif'
            with rasterio.open(output, 'w', **profile) as tif2:
                tif2.write(image, 1 )

        return image

    def raster_normalize(self):
        '''
        ### Raster Normalization by mean and std
            Input: a raster to apply normalization
        '''
        with rasterio.open(self.__imgdir, 'r+') as tif:
            image = tif.read(1) 
            profile = tif.meta.copy()
            profile.update({'nodata': np.nan})

            # Check if exist nodata
            if np.isnan(np.sum(image)) != True:
                # Using 1st value as nodata value and replacing by np.nan
                wrong_nodata = image[0][0]
                image[np.where(image == wrong_nodata)] = np.nan

            # Getting info to compute Normalization
            mean_ = np.nanmean(image)
            std_ = np.nanstd(image)
            normalized = (image-mean_)/std_

        # Saving
        if self.__write == True:
            output = self.__imgdir[:-4] + '_Normalized.tif'
            with rasterio.open(output, 'w', **profile) as tif2:
                tif2.write(normalized, 1 )
        
        return normalized

    def raster_standartize (self):
        '''
        ### Raster Standartize by min and max
            Input: a raster to statndartize
        '''
        with rasterio.open(self.__imgdir) as tif:
            new_tif = tif.read(1) 
            profile = tif.profile.copy()
            profile.update({'nodata': np.nan})

            # Check if exist nodata
            if np.isnan(np.sum(new_tif)) != True:
                # Using 1st value as nodata value and replacing by np.nan
                wrong_nodata = new_tif[0][0]
                new_tif[np.where( new_tif==wrong_nodata)] = np.nan

            # Getting info to compute  Standartize
            max_ = np.nanmax(new_tif)
            min_ = np.nanmin(new_tif)

            pradonizado = (new_tif-min_)/(max_ - min_)

        # Saving
        if self.__write == True:
            output = self.__imgdir[:-4] + '_Stand.tif'
            with rasterio.open(output, 'w', **profile) as tif2:
                tif2.write(pradonizado, 1 )

        return pradonizado

    def raster_to_shp_points(self):
        '''
        ### Transform a raster to shapefile points by the pixel centroid
            Input: a raster path to write a new shapefile
            
        '''
        # Loading raster
        with rasterio.open(self.__imgdir) as tif:
            image = tif.read(1)  
            transform = tif.transform
            epsg = tif.profile['crs']

            # Check if exist nodata
            if np.isnan(np.sum(image)) != True:
            # Using 1st value as nodata value
                wrong_nodata = image[0][0]
                # Replacing it by np.nan
                image[np.where( image==wrong_nodata)] = np.nan
       
        # Getting XY position and values from raster
        points = []
        for (i, j), value in np.ndenumerate(image):           
            # Skip nan values to only compute the study area
            if np.isnan(value) != True:

                # Getting pixel centroid
                x, y = transform * (j + 0.5, i + 0.5)
                
                # Saving into a tuple (why not a dictionary?)
                point = (x, y, value)
                points.append(point)

        # Reading tuple as a pandas DataFrame
        df = pd.DataFrame(points, columns=['X', 'Y', 'value'])
        
        # Creating a Geometry and dropping X, Y columns
        geometry = [Point(xy) for xy in zip(df.X, df.Y,)]
        df = df.drop(['X', 'Y'], axis=1)

        # Creating a geodataframe and saving it
        gdf_ = gp.GeoDataFrame(df, crs={'init' : str(epsg)}, geometry=geometry)

        # Exporting shapefile
        if self.__write == True:
            out_shp = self.__imgdir[:-4] + '.shp'
            gdf_.to_file(out_shp )

        return gdf_

    def polygonize(self):
        with rasterio.open(self.__imgdir, 'r+') as tif:
            tif.nodata = np.nan
            epsg = str(tif.crs)
            rast = tif.read(1, masked=True)

            results = ( {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) in enumerate(
                rasterio.features.shapes(rast, transform=tif.transform)))

        geoms = list(results)

        # Creating geodataframe and setting CRS
        polygonized = gp.GeoDataFrame.from_features(geoms)
        polygonized.crs = {'init': epsg}

        return polygonized


class Raster_resample():
    '''
    ### Resampling raster by mean
    Input: raster, scale and base raster
    '''
    def __init__(self, raster_dir, raster_base='', scale=0):
        self.__raster = raster_dir
        self.__scale = scale
        self.__raster_base = raster_base

    def resample_by_scale(self):
        '''
        ### Resampling raster using a scale
        #     Input: raster directory and scale
        '''
        if self.__raster_base == 0:
            print('Scale must be greater than 0')
            pass
        else:
            with rasterio.open(self.__raster, 'r+') as tif:
                tif.nodata = np.nan
                tif_profile = tif.meta.copy()
                tif_transf = tif.transform
                print(f'Original raster: \n {tif_transf}')
                
                # Raster rescaling
                transform = rasterio.Affine( round(tif_transf.a * self.__scale), tif_transf.b, tif_transf.c, 
                                            tif_transf.d, round(tif_transf.e * self.__scale), tif_transf.f)
                print(f'Transformed raster: \n {transform}')

                # Computing new heigh and width
                height = int((tif.height) / self.__scale )
                width = int((tif.width)/ self.__scale )

                # Updating profile with new info
                tif_profile.update(transform=transform, driver='GTiff', height=height, width=width, crs=tif.crs,
                                    count = tif.count)

                # Reading raster to resample it
                data = tif.read(
                        out_shape=(int(tif.count), int(height), int(width)),
                        resampling=rasterio.enums.Resampling.average)
                
            # Writing a new raster
            spatial_resolution = round(tif_transf.a * self.__scale)
            output = self.__raster[:-4] + f'_R{spatial_resolution}.tif'
            
            with rasterio.open(output, 'w', **tif_profile) as dst:
                dst.write(data)
                    
            return data                

    def resample_by_raster(self):
        '''
        ### Resampling raster by another raster
            Input: two rasters directories (one to be resampled and another for base)
        '''   
        with rasterio.open(self.__raster_base) as base:
            profile = base.meta.copy()
            height = base.shape[0]
            width = base.shape[1]

            # Resolution output image transform
            xres = int((base.bounds.right  - base.bounds.left) /width)
            yres = int((base.bounds.top  - base.bounds.bottom ) / height )

            # Affine
            transform = rasterio.Affine(xres, base.transform.b, base.transform.c,
                                        base.transform.d, -yres, base.transform.f)

            # Getting the original raster profile and updating with new information
            profile.update(transform=transform, driver='GTiff', height=height, width=width, 
                            crs=base.crs, count=base.count, nodata= np.nan, dtype='float32' )
                                
        with rasterio.open(self.__raster, 'r+') as tif:
            # Reading raster to resample it
            data = tif.read(out_shape=(int(tif.count), int(height), int(width)),
                            resampling=rasterio.enums.Resampling.average)

            # Writing a new raster
            output = self.__raster[:-4] + f'_R{xres}_.tif'

        with rasterio.open(output, 'w', **profile) as dst:
            dst.write(data)

        return data
        # Resource: #https://gis.stackexchange.com/questions/329434/creating-an-in-memory-rasterio-dataset-from-numpy-array?rq=1


class Shape_operations():

    def __init__(self, pathshp1, pathshp2):
        self.__path_shp1 = pathshp1
        self.__path_shp2 = pathshp2

    def clip_shapes(self):
        '''
        ### Compute intersection operation (as in GeoPandas/QGIS)
            Input: two shapefiles directories
            Output: a new shapefile from common areas in two shapes
        '''
        # Reading shapefiles
        shp1 = gp.read_file(self.__path_shp1 )
        shp2 = gp.read_file(self.__path_shp2)

        # Check crs
        crs1, crs2 = shp1.crs, shp2.crs
        if crs1 == crs2:
            # Clipping shapefiles
            result = gp.overlay(shp1, shp2, how='intersection')
            result = result.drop('DN', axis=1)

            # Saving shapefile
            output_name = self.__path_shp1 [:-4] + '_rec10m.shp'
            result.to_file(self.__path_shp1 + output_name)

            info_newshp = dict( {'columns names': result.columns,
                                'shp1 extent': shp1.total_bounds,
                                'shp2 extent': shp2.total_bounds,
                                'final extent': result.total_bounds} )
        
        else:
            print('Shapefiles with different EPSG')

        return info_newshp

    def crs_change(self, epsg):
        '''
        ### Change shapefile EPSG
            Input: one shapefile direcotory and the desired EPSG
        '''
        # Reading shapefile
        shp1 = gp.read_file(self.__path_shp1 )
        
        # Changing EPSG
        shp1.crs = {'init': str(epsg)}

        # Saving
        output_name = self.__path_shp1 [:-4] +  str(epsg) + '.shp'
        shp1.to_file(output_name)

  
class UAV_funcs():

    def __init__(self, img_directory):
        self.__imgdir = img_directory
        # self.__shpdir = shp_directory

    def band_normalized_t1(self):
        '''
        ### Execute band normalization
            Input: a raster directory with 3 bands (R,G,B) 
            Output: will be a raster per band divided by sum of them
        '''
        with rasterio.open(self.__imgdir, 'r+') as tif:

            # Reading profile and Setting nan values
            tif.nodata = np.nan
            profile = tif.meta.copy() 
            profile.update({'count': 1, 'compress': 'lzw', 'dtype': 'float32', 'Nodata': np.nan})

            # Checking bands:
            band_info = tif.indexes

            # Creating names for output
            outputR = self.__imgdir[:-4] +  '_R_N.tif'
            outputG = self.__imgdir[:-4] +  '_G_N.tif'
            outputB = self.__imgdir[:-4] +  '_B_N.tif'

            # Reading raster by tiles (raster windows)
            # tiles = tif.block_windows(1)
            for band in band_info:
                if band == 1:
                    with rasterio.open(outputR, 'w', **profile) as dst:
                        tiles = tif.block_windows(1)

                        for idx, window in tqdm(tiles):   
                            band_R = tif.read(1, window=window, masked=True).astype('float32')             
                            band_G = tif.read(2, window=window, masked=True).astype('float32')
                            band_B = tif.read(3, window=window, masked=True).astype('float32')
                
                            # Como resolver o problema do 0?
                            imgR = band_R / (band_R + band_G + band_B)
                            dst.write_band(1, imgR, window=window)

                elif band == 2:
                    tiles = tif.block_windows(1)
                    
                    with rasterio.open(outputG, 'w', **profile) as dst:
                        for idx, window in tqdm(tiles):
                            band_R = tif.read(1, window=window, masked=True).astype('float32')             
                            band_G = tif.read(2, window=window, masked=True).astype('float32')
                            band_B = tif.read(3, window=window, masked=True).astype('float32')
                            imgG = band_G / (band_R + band_G + band_B) 
                            dst.write_band(1, imgG, window=window)

                if band == 3:
                    tiles = tif.block_windows(1)

                    with rasterio.open(outputB, 'w', **profile) as dst:
                        for idx, window in tqdm(tiles):   
                            band_R = tif.read(1, window=window, masked=True).astype('float32')             
                            band_G = tif.read(2, window=window, masked=True).astype('float32')
                            band_B = tif.read(3, window=window, masked=True).astype('float32')
                            imgB = band_B / (band_R + band_G + band_B) 
                            dst.write_band(1, imgB, window=window)

        return [imgR, imgG, imgB]           

    def band_normalized_t2(self):
        '''
        ### Execute band normalization
            Input: a raster directory with 3 bands (R,G,B) 
            Output: will be ONE RASTER with each band divided by sum of them
        '''

        with rasterio.open(self.__imgdir, 'r+') as tif:

            # Reading profile and Setting nan values
            tif.nodata = np.nan
            profile = tif.meta.copy() 
            profile.update({'compress': 'lzw', 'dtype': 'float32', 'Nodata': np.nan})

            # Checking bands:
            band_info = tif.indexes

            # Creating names for output
            output = self.__imgdir[:-4] +  '_N_.tif'

            # Reading raster by tiles (raster windows)
            tiles = tif.block_windows(1)

            with rasterio.open(output, 'w', **profile) as dst:

                for idx, window in tqdm(tiles):   
                    band_R = tif.read(1, window=window, masked=True).astype('float32')             
                    band_G = tif.read(2, window=window, masked=True).astype('float32')
                    band_B = tif.read(3, window=window, masked=True).astype('float32')
        
                    imgR = band_R / (band_R + band_G + band_B)
                    imgG = band_G / (band_R + band_G + band_B)
                    imgB = band_B / (band_R + band_G + band_B)
                    result = np.array([imgR, imgG, imgB])

                    dst.write(result, window=window)

        return result


class Outliers_check():
    '''
    ### Find/Remove an outlier which is greater or less than 3*std+mean
    Input: raster, array with outliers
    Additional criteria: write file or not
    '''
    def __init__(self, path_img='', outliers='None', writing=False, dataset=''):
        self.__pathimg = path_img
        self.__write = writing
        self.__outliers = outliers
        self.__data = dataset

    def find_outlier(self):
        '''
        ### Look for outliers above 3*std
        '''
        with rasterio.open(self.__pathimg) as tif:
            data = tif.read(1)
            profile = tif.meta.copy()
            
        # Remove nan values
        data = data[np.logical_not(np.isnan(data))]

        # Searching for outliers
        cut_off = np.std(data) * 3
        lower_limit  = np.mean(data) - cut_off 
        upper_limit = np.mean(data) + cut_off

        # Selecting them 
        outliers = data[np.where((data < lower_limit) | (data > upper_limit))]
        outliers = np.unique(outliers)

        return outliers

    def remove_outlier(self):
        if self.__outliers == 'None':
            print('Inform array with outliers')
            pass
        else:
            with rasterio.open(self.__pathimg) as tif:
                # Reading data and copy
                data2 = tif.read(1)
                profile = tif.meta.copy()

            # Look for outliers
            for i, j in np.ndindex((data2.shape[0]-1, data2.shape[1]-1)):    
                if data2[i,j] in self.__outliers:

                    # Replacing them by mean in a (3,3) array
                    r = np.zeros((3,3))
                    r = data2[i-1: i+2, j-1: j+2]
                    data2[i,j] = np.nanmean(r)

            if self.__write == True:
                output = self.__pathimg[:-4] + '_LessOut.tif'
                with rasterio.open(output, 'w', **profile) as dst:
                    dst.write(data2, 1)      
                
            return data2 

    def normality_check(self):
        '''
        ### Description:
        To check if a raster has gaussian distribution you can pass a .tif or
        a numpy 3D array
        '''
        if os.path.exists(self.__pathimg): 
            with rasterio.open(self.__pathimg) as tif:
                # Reading data and copy
                data = tif.read(1)
                profile = tif.meta.copy()

        else:
            data = self.__data 
        
        data = data[np.logical_not(np.isnan(data))]

        # Shapiro-Wilk, D'Agostino and Kolmogorov-Sirmov tests
        shapiro_wilk = scipy.stats.shapiro(data)
        agostino = scipy.stats.normaltest(data)
        kolmogorov = scipy.stats.kstest(data, 'norm')
        
        tests = [shapiro_wilk, agostino, kolmogorov]
        pvalues = []
        for test in tests:
            if test[1] < 0.05:
                pvalue = 'non-normal'
            else:
                pvalue = 'normal'

            pvalues.append(pvalue)
        
        result = {'shapiro': (shapiro_wilk[0], shapiro_wilk[1], pvalues[0]), 
                    'Agostino': (agostino[0], agostino[1], pvalues[1]), 
                    'Kolmogorov': (kolmogorov[0], kolmogorov[1], pvalues[2])}

        return result

