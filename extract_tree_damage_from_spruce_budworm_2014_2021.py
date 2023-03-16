#coding: utf-8

"""
Summary
-------
Script to produce predictions of SBW outbreak extent using RF modelling and concave hulls. 

"""

import geopandas as gpd
import pandas as pd 
from geopandas.tools import sjoin
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import shape
from shapely.geometry.multipolygon import MultiPolygon
from descartes import PolygonPatch
import time
import math
import scipy.stats as stats
import numpy as np
import os, sys
from pyproj import CRS, Transformer
import fiona
import statsmodels.api as sm
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
import sklearn
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.inspection import permutation_importance

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib as mpl
from osgeo import ogr, gdal,osr
from math import floor

from pygam import GAM
from pygam import LogisticGAM, s, f, te, l
from shapely.ops import unary_union

import warnings
warnings.filterwarnings('ignore')

import alphashape

def concave_hull(points,a,compar):
    #You aren't going to run it now, you're going to run it later, after classifying sat image
    alpha = a #for concave hull
        
    print('yes')
    hull = alphashape.alphashape(points,a) 

    print('check!') 
    
    try: 
        if a != 0:  
            hull = MultiPolygon(hull) #[]
        else:
            hull = MultiPolygon([hull])
    except TypeError:
        hull = MultiPolygon([hull])

    if len(hull) != 0:

        p1 = gpd.GeoDataFrame(index=[0], crs='ESRI:102001', geometry=[hull])
        hull_explode = p1.explode(ignore_index=True)
        
        
        polygon = gpd.GeoDataFrame(crs='ESRI:102001', geometry=list(hull_explode['geometry']))
        print(polygon.head(10))
        polygon['AREA'] = (polygon['geometry'].area) *0.000001
        print(polygon.head(10))
        polygon1 = polygon[polygon['AREA'] >= 3]

        if len(polygon1) == 0:
            polygon = polygon[polygon['AREA'] >= 0]
        else:
            polygon = polygon1

        return polygon
    else:
        print('no geometry')
        return []

def duplicated(yeari,mod):

    year = str(yeari)
    yearp = str(1998)


    files = ['100m_on_ndmi_102001_'+year,'asm_'+year,'buff8_'+year,\
             '100m_on_ndmi_102001_'+yearp,'100m_on_nbr1_102001_'+year,\
             '100m_on_nbr1_102001_'+yearp,'100m_on_b4b5_102001_'+year,'100m_on_b4b5_102001_'+yearp]
    names = ['ndmi_'+year,'dam','small_buff','ndmi_'+yearp,\
             'nbr1_'+year,'nbr1_'+yearp,'b4b5_'+year,'b4b5_'+yearp] 
    pred = {}
    transformers = []
    cols_list = []
    rows_list = [] 

    for fi,n in zip(files,names): 
        print(fi)
        file_name_raster = fi
        src_ds = gdal.Open('rasters/new_res/final/'+file_name_raster+'.tif')
        rb1=src_ds.GetRasterBand(1)
        cols = src_ds.RasterXSize
        cols_list.append(cols)
        rows = src_ds.RasterYSize
        rows_list.append(rows) 
        data = rb1.ReadAsArray(0, 0, cols, rows)
        print('Success in reading file.........................................') 
        pred[n] = data.flatten()
        print(len(data.flatten()))
        transform=src_ds.GetGeoTransform()
        transformers.append(transform)
    
    #pred['age'] = pred['age'] + (int(year)-2011)

    #Reverse of before

    pred['diff'] = pred['ndmi_'+year] - pred['ndmi_'+yearp]
    pred['nbr1_diff'] = pred['nbr1_'+year] - pred['nbr1_'+yearp]
    pred['b4b5_diff'] = pred['b4b5_'+year] - pred['b4b5_'+yearp]


    col_num = cols_list[0]
    row_num = rows_list[0]
    ulx, xres, xskew, uly, yskew, yres  = transformers[0]
    print(transformers[0])
    lrx = ulx + (col_num * xres)
    lry = uly + (row_num * yres)
    print(lrx)
    print(lry)

    Yi = np.linspace(np.min([uly,lry]), np.max([uly,lry]), row_num)
    Xi = np.linspace(np.min([ulx,lrx]), np.max([ulx,lrx]), col_num)
    

    Xi, Yi = np.meshgrid(Xi, Yi)
    Xi, Yi = Xi.flatten(), Yi.flatten()

    X_reshape = Xi.reshape(row_num,col_num)[::-1]
    Xi = X_reshape.flatten()
    Y_reshape = Yi.reshape(row_num,col_num)[::-1]
    Yi = Y_reshape.flatten()
    print(len(Xi))
    print(Xi[0]) 
  
    pred['lon'] = Xi
    pred['lat'] = Yi

    
    df = pd.DataFrame(pred).dropna(how='any') #.iloc[::100, :]
    
    #df = df[df['combo'] > 0] #exclude no species

    df_calc = df 

    print(len(df))
    df = df[df['ndmi_'+year] >= -1]

    df = df[df['ndmi_'+year] <= 1]
    print(len(df))
    df = df[df['ndmi_'+yearp] >= -1]
    print(len(df))
    df = df[df['ndmi_'+yearp] <= 1]
    print(len(df))

    p95 = np.percentile(df_calc['b4b5_'+yearp], 99)
    print(p95)
    p5 = np.percentile(df_calc['b4b5_'+yearp], 1)
    print(p5)
    df = df[df['b4b5_'+yearp] >= p5]
    df = df[df['b4b5_'+yearp] <= p95]

    p95 = np.percentile(df_calc['b4b5_'+year], 99)
    print(p95)
    p5 = np.percentile(df_calc['b4b5_'+year], 1)
    print(p5)
    df = df[df['b4b5_'+year] >= p5]
    df = df[df['b4b5_'+year] <= p95]

    p95 = np.percentile(df_calc['nbr1_'+yearp], 99)
    print(p95)
    p5 = np.percentile(df_calc['nbr1_'+yearp], 1)
    print(p5)
    df = df[df['nbr1_'+yearp] >= p5]
    df = df[df['nbr1_'+yearp] <= p95]

    p95 = np.percentile(df_calc['nbr1_'+year], 99)
    print(p95)
    p5 = np.percentile(df_calc['nbr1_'+year], 1)
    print(p5)
    df = df[df['nbr1_'+year] >= p5]
    df = df[df['nbr1_'+year] <= p95]


 
    df['dam'] = np.where(df['dam'] >= 1,1,0)
    print(df['dam'].head(10))
    df = df.iloc[::10, :]
    df_save = df
    
    lengths = []

    trainer = [] 
    for cl in [1,0]: #used to have 3 
        df_f = df[df['dam'] == cl].dropna(how='any')
        if cl != 0:
            #num = int(len(df_f) / 1000)
            if len(df_f) >= 2000: 
                num = 2000
                trainer.append(df_f.sample(n=num,random_state=1)) #500000
                lengths.append(num)
            else:
                print('Check - less than 2000')
                trainer.append(df_f)
        else:
            #number of negatives varies with cloud mask, etc.
            df_f = df_f[df_f['small_buff'] != 1]
            #num = int(len(df_f) / 1000)
            num = 2000
            trainer.append(df_f.sample(n=num,random_state=1))
            

    df2 = pd.concat(trainer)
    df2 = df2.reset_index(drop=True).dropna(how='any')


    df_trainX = df2[['ndmi_'+year,'nbr1_'+year,'b4b5_'+year,\
                     'diff','nbr1_diff','b4b5_diff']] #Actually for GAM it has to be an array
    X = np.array(df2[['ndmi_'+year,'nbr1_'+year,'b4b5_'+year,\
                     'diff','nbr1_diff','b4b5_diff']]) #,'lat','lon'


    df_trainY = np.array(df2[['dam']]).reshape(-1, 1)
    Y = np.array(df2['dam'])

    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import confusion_matrix
    count = 0 


    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=1)
    
    mattc = []
    from sklearn.metrics import matthews_corrcoef
    for train_index, test_index in sss.split(X, Y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        print(len(X_train))

        bestF = CV_rfc.fit(X, Y)
        
        Ztest = mod.predict(X_test)
        print(matthews_corrcoef(y_test, Ztest))
        mattc.append(matthews_corrcoef(y_test, Ztest))
        

    df_save['tracker'] = list(range(0,len(df_save))) #index
    rem_track = df_save.dropna(how='any')

    Zi = mod.predict(np.array(rem_track[['ndmi_'+year,'nbr1_'+year,'b4b5_'+year,\
                     'diff','nbr1_diff','b4b5_diff']]))

    rem_track['pred'] = Zi

    matt = []

    rep = np.array(rem_track['dam'])

    Zn = np.array(rem_track['pred'])
        
    print(confusion_matrix(rep, Zn))


    from sklearn.metrics import matthews_corrcoef
    print(matthews_corrcoef(rep, Zn))
    matt.append(matthews_corrcoef(rep, Zn))

    index_max = max(range(len(matt)), key=matt.__getitem__)

    list_thresh = [0.5]
    thresh = list_thresh[index_max]
    print(thresh)

    add_track = df_save[pd.isnull(df_save).any(axis=1)]
    add_track['pred'] = -9999

    total = pd.concat([rem_track,add_track])
    total = rem_track


    mort = rem_track[rem_track['pred'] == 1]
    
    points = [(x,y,) for x, y in zip(mort['lon'],mort['lat'])]
    na_map3 = na_map[na_map['DAM'] >=2]
    
    ch2 = concave_hull(points,0.0141,na_map3)
    print(ch2.head(10))
    

    if len(ch2) > 0: 
        ch2.plot(ax=ax[1],facecolor='None',edgecolor='k')

    if len(ch2) > 0:

        ch2.to_file('rasters/concave_hull/'+str(yeari)+'_test_y2013_feb3.shp', driver='ESRI Shapefile')
        
if __name__ == "__main__":

    year = str(2021)
    yearp = str(2013)


    files = ['100m_on_ndmi_102001_'+year,'asm_'+year,'combo','buff8_'+year,'age','elev','soil_text',\
             '100m_on_ndmi_102001_'+yearp,'100m_on_nbr1_102001_'+year,\
             '100m_on_nbr1_102001_'+yearp,'100m_on_b4b5_102001_'+year,'100m_on_b4b5_102001_'+yearp,]
    names = ['ndmi_'+year,'dam','combo','small_buff','age','elev', 'soil_text','ndmi_'+yearp,\
             'nbr1_'+year,'nbr1_'+yearp,'b4b5_'+year,'b4b5_'+yearp] 
    pred = {}
    transformers = []
    cols_list = []
    rows_list = [] 

    for fi,n in zip(files,names): 
        print(fi)
        file_name_raster = fi
        src_ds = gdal.Open('rasters/new_res/final/'+file_name_raster+'.tif')
        rb1=src_ds.GetRasterBand(1)
        cols = src_ds.RasterXSize
        cols_list.append(cols)
        rows = src_ds.RasterYSize
        rows_list.append(rows) 
        data = rb1.ReadAsArray(0, 0, cols, rows)
        print('Success in reading file.........................................') 
        pred[n] = data.flatten()
        print(len(data.flatten()))
        transform=src_ds.GetGeoTransform()
        transformers.append(transform)
    
    pred['age'] = pred['age'] + (int(year)-2011)

    pred['diff'] = pred['ndmi_'+yearp] - pred['ndmi_'+year]
    pred['nbr1_diff'] = pred['nbr1_'+yearp] - pred['nbr1_'+year]
    pred['b4b5_diff'] = pred['b4b5_'+yearp] - pred['b4b5_'+year]


    col_num = cols_list[0]
    row_num = rows_list[0]
    ulx, xres, xskew, uly, yskew, yres  = transformers[0]
    print(transformers[0])
    lrx = ulx + (col_num * xres)
    lry = uly + (row_num * yres)
    print(lrx)
    print(lry)

    Yi = np.linspace(np.min([uly,lry]), np.max([uly,lry]), row_num)
    Xi = np.linspace(np.min([ulx,lrx]), np.max([ulx,lrx]), col_num)
    

    Xi, Yi = np.meshgrid(Xi, Yi)
    Xi, Yi = Xi.flatten(), Yi.flatten()

    X_reshape = Xi.reshape(row_num,col_num)[::-1]
    Xi = X_reshape.flatten()
    Y_reshape = Yi.reshape(row_num,col_num)[::-1]
    Yi = Y_reshape.flatten()
    print(len(Xi))
    print(Xi[0]) 
  
    pred['lon'] = Xi
    pred['lat'] = Yi

    
    df = pd.DataFrame(pred).dropna(how='any') 
    

    df_calc = df 

    print(len(df))
    df = df[df['ndmi_'+year] >= -1]

    df = df[df['ndmi_'+year] <= 1]
    print(len(df))
    df = df[df['ndmi_'+yearp] >= -1]
    print(len(df))
    df = df[df['ndmi_'+yearp] <= 1]
    print(len(df))

    p95 = np.percentile(df_calc['b4b5_'+yearp], 99)
    print(p95)
    p5 = np.percentile(df_calc['b4b5_'+yearp], 1)
    print(p5)
    df = df[df['b4b5_'+yearp] >= p5]
    df = df[df['b4b5_'+yearp] <= p95]

    p95 = np.percentile(df_calc['b4b5_'+year], 99)
    print(p95)
    p5 = np.percentile(df_calc['b4b5_'+year], 1)
    print(p5)
    df = df[df['b4b5_'+year] >= p5]
    df = df[df['b4b5_'+year] <= p95]

    p95 = np.percentile(df_calc['nbr1_'+yearp], 99)
    print(p95)
    p5 = np.percentile(df_calc['nbr1_'+yearp], 1)
    print(p5)
    df = df[df['nbr1_'+yearp] >= p5]
    df = df[df['nbr1_'+yearp] <= p95]

    p95 = np.percentile(df_calc['nbr1_'+year], 99)
    print(p95)
    p5 = np.percentile(df_calc['nbr1_'+year], 1)
    print(p5)
    df = df[df['nbr1_'+year] >= p5]
    df = df[df['nbr1_'+year] <= p95]

 
    df['dam'] = np.where(df['dam'] >= 2,1,0)
    df_save = df
    
    lengths = []

    trainer = [] 
    for cl in [1,0]: #used to have 3 
        df_f = df[df['dam'] == cl].dropna(how='any')
        if cl != 0:

            num = 2000
            trainer.append(df_f.sample(n=num,random_state=1)) #500000
            lengths.append(num)
        else:
            #number of negatives varies with cloud mask, etc.
            df_f = df_f[df_f['small_buff'] != 1]
            num = 2000
            trainer.append(df_f.sample(n=num,random_state=1))
            


    df2 = pd.concat(trainer)
    df2 = df2.reset_index(drop=True).dropna(how='any')


    df_trainX = df2[['ndmi_'+year,'nbr1_'+year,'b4b5_'+year,\
                     'diff','nbr1_diff','b4b5_diff']] 
    X = np.array(df2[['ndmi_'+year,'nbr1_'+year,'b4b5_'+year,\
                     'diff','nbr1_diff','b4b5_diff']])


    df_trainY = np.array(df2[['dam']]).reshape(-1, 1)
    Y = np.array(df2['dam'])

    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import confusion_matrix
    count = 0 

    rfc = RandomForestClassifier(random_state=1)
##    param_grid = { 
##    'max_depth': [5, 10, 30],
##    'max_features': ['sqrt'],
##    'min_samples_leaf': [1,3,5],
##    'min_samples_split': [2,20,40,60]
##    }

    param_grid = { 
    'max_depth': [30],
    'max_features': ['sqrt'],
    'min_samples_leaf': [5],
    'min_samples_split': [20]
    }
    
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5) #5
    bestF = CV_rfc.fit(X, Y)
    print(CV_rfc.best_params_)

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=1)
    
    mattc = []
    from sklearn.metrics import matthews_corrcoef
    for train_index, test_index in sss.split(X, Y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        print(len(X_train))

        bestF = CV_rfc.fit(X, Y)
        
        Ztest = bestF.predict(X_test)
        print(matthews_corrcoef(y_test, Ztest))
        mattc.append(matthews_corrcoef(y_test, Ztest))
        

    df_save['tracker'] = list(range(0,len(df_save))) #index
    rem_track = df_save.dropna(how='any')

    Zi = bestF.predict(np.array(rem_track[['ndmi_'+year,'nbr1_'+year,'b4b5_'+year,\
                     'diff','nbr1_diff','b4b5_diff']]))

    rem_track['pred'] = Zi

    matt = []

    rep = np.array(rem_track['dam'])

    Zn = np.array(rem_track['pred'])
        
    print(confusion_matrix(rep, Zn))

    from sklearn.metrics import matthews_corrcoef
    print(matthews_corrcoef(rep, Zn))
    matt.append(matthews_corrcoef(rep, Zn))

    index_max = max(range(len(matt)), key=matt.__getitem__)

    list_thresh = [0.5]
    thresh = list_thresh[index_max]
    print(thresh)

    add_track = df_save[pd.isnull(df_save).any(axis=1)]
    add_track['pred'] = -9999

    total = pd.concat([rem_track,add_track])
    total = rem_track



    mort = rem_track[rem_track['pred'] == 1]
    
    points = [(x,y,) for x, y in zip(mort['lon'],mort['lat'])]
    na_map3 = na_map[na_map['DAM'] >=2]
    
    ch2 = concave_hull(points,0.0141,na_map3)
    

    if len(ch2) > 0: 
        ch2.to_file('rasters/concave_hull/2021_test_y2013_feb1.shp', driver='ESRI Shapefile')

    yfor = list(range(2014,2020+1))
    for y in yfor: 

        duplicated(y,bestF)

