import xarray as xr
import rioxarray as rxr
import numpy as np
import stackstac
import os
import leafmap
import requests
from tqdm import tqdm
import pystac_client
import planetary_computer
from matplotlib import pyplot as plt


def preprocess_data(bbox, start, end, index_name="RVI", scale=2, interval="1Q", deseason=False):
    """
    This function returns a resampled time series of radar vegetation index values using microsoft planetary computer API

    Inputs:
    --------
    bbox : tuple 
        containing min x, min y, max x, and max y of bounding box in geographic coordinates (degrees)
    start : string
        start date for search
    end : string
        end date for search
    index_name : string
        name of index. Valid options: "mRFDI","RVI","mRVI","CR","BR". Default is "RVI"
    scale : intiger
        scale used to coarsen pixel size. Default 2
    interval : string
        interval to resample to. Default is 1Q (3 months)
    deseason : boolean
        if True, output units are anomalies; if False, output units are raw index values. Default is False
        
    Returns:
    --------
    composite : dask dataArray 
        time series of radar vegetation index values
    """
    search = get_sar_pc(bbox, start, end)
    stack = items_to_xarray(search, bbox)
    index = calculate_index(stack, index_name)
    coarsened = coarsen_image(index, scale)
    composite = temporal_composite(coarsened, interval)
    if deseason==True:
        return deseason_quarter(composite)
    else:
        return composite



    
def get_sar_pc(bbox, start, end):
    """
    This function returns Sentinel 1 SAR items from Planetary Computer API

    Inputs:
    --------
    bbox : tuple 
        containing min x, min y, max x, and max y of bounding box in geographic coordinates (degrees)
    start : string
        start date for search
    end : string
        end date for search

    Returns:
    --------
    items : STAC feature collection
        time series of radar vegetation index values
    """
    catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",  # connect to Planetary Computer API
    modifier=planetary_computer.sign_inplace,               
    )

    search = catalog.search(filter_lang="cql2-json", filter={
        "op": "and",
        "args": [{"op": "s_intersects", "args": [{"property": "geometry"}, bbox]},
                 {"op": "=", "args": [{"property": "collection"}, "sentinel-1-rtc"]},
                 {"op": "anyinteracts", "args": [{"property": "datetime"}, f"{start}/{end}"]},
                 {"op": "=", "args": [{"property": "sar:polarizations"}, ["VV", "VH"]]},
                 {"op": "=", "args": [{"property": "sat:orbit_state"}, "ascending"]},
                ]
    }
                           )

    items = search.get_all_items()  # get items into new variable
    return items


def items_to_xarray(items, aoi):
    """
    This function takes items from STAC search and loads them into dask dataarray 

    Inputs:
    --------    
    items : STAC feature collection
        time series of radar vegetation index values
    aoi : tuple 
        containing min x, min y, max x, and max y of bounding box in geographic coordinates (degrees)

    Returns:
    --------
    stack : Dask xarray.DataArray
        time series of SAR observations, 10m resolution, with UTM 37S projection
    """
    stack = stackstac.stack(
        items,
        epsg=32737,
        resolution=10,
        bounds_latlon=aoi,
    )
    
    return stack

def coarsen_image(stack, scale = 2):
    """
    This function coarsens the pixel size of a xarray dataarray object

    Inputs:
    --------    
    stack : Dask xarray.DataArray
        time series of SAR observations
    scale : int
        scalar to coarsen image to. output resolution will be scale * 10. Default is 2 (20 meter pixel)

    Returns:
    --------
    stack : Dask xarray.DataArray
        time series of SAR observations with coarsened pixel size
    """
    coarsened = stack.coarsen(x=scale, y=scale, boundary='pad').mean()
    return coarsened

def temporal_composite(stack, interval = "1Q"):
    """
    This function resamples a dataarray time series.

    Inputs:
    --------    
    stack : Dask xarray.DataArray
        time series of SAR observations
    interval : string
        interval to resample to. Default 1Q (3 months)

    Returns:
    --------
    stack : Dask xarray.DataArray
        time series of SAR observations with resampled time step
    """
    composite = stack.resample(time=interval).mean("time")

    return composite

def calculate_index(stack, index):
    """
    This function calculates a radar index from VV and VH bands

    Inputs:
    --------    
    stack : Dask xarray.DataArray
        time series of SAR observations
    index : string
        name of index to use. Valid options: "mRFDI","RVI","mRVI","CR","BR". Default is "RVI"

    Returns:
    --------
    stack : Dask xarray.DataArray
        time series of index value
    """
    vh, vv = stack.sel(band = "vh"), stack.sel(band = "vv")

    if index == "mRFDI":
        value = (vv-vh)/(vv+vh)
        return value
    elif index == "RVI":
        value = (4*vh)/(vv+vh)
        return value
    elif index == "mRVI":
        value = ((vv/(vv+vh))**0.5)+(4*vh/(vv+vh)) 
        return value
    elif index == "CR":
        value = (vh-vv)
        return value
    elif index == "BR":
        value = (vh/vv)
        return value
    else:
        print("Invalid index name")
def img_change(stack):
    """
    Calculating change from previous image to current image

    time[0] will be NA
    
    """
    shifted = stack.shift(time = 1)
    change = (stack-shifted)
    return change
    
def deseason_func(x):
    
    diff = x - x.mean(dim='time')
    return diff
    
def deseason_quarter(stack):
    """

    """
    deseason = stack.groupby(stack.time.dt.quarter).map(deseason_func)
    return deseason
    
def linear_trend(stack):
    """
    Function for calculating trend
    
    """
    stack.coords['year'] = ('time', stack.time.dt.year.data)
    trend = stack.polyfit(dim = "year", deg = 1)
    return trend

def forest_cover_plot(stack, threshold, RFDI=False):

    if RFDI==True:
        stack["forest"] = stack < threshold
    else:
        stack["forest"] = stack > threshold

    forestpixels=stack.forest.sum(dim = ['y', 'x'])

    fig, ax = plt.subplots()
    forestpixels.plot(ax=ax)
    plt.ylabel("N Pixels Classified as Forest")
    plt.title("Forest Cover in AOI")
    return ax


def timeseries_plot(stack, index):
    tsmean = stack.mean(dim = ['y', 'x'], skipna = True)
    fig, ax = plt.subplots()
    tsmean.plot(ax=ax)
    plt.ylabel(f"{index}")
    plt.title(f"Vegetation Index time series")
    return ax

def plot_year(stack, year):
    return stack.sel(time=year).plot.imshow(col="time", col_wrap=4)

def linear_reg_10yr(stack):
    years = ('2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024')

    for i in range(0, 10):
        x = stack.sel(time=years[i])
        x.coords['month'] = ('time', x.time.dt.month.data)
        trend = x.polyfit(dim = "month", full = True, deg = 1)
        linecoeff = trend.sel(degree = 1)
        fig, ax = plt.subplots(ncols=2, figsize=(10,2))
        linecoeff.polyfit_coefficients.plot.imshow(ax=ax[0])
        trend.polyfit_residuals.plot.imshow(ax=ax[1])    
        ax[0].set_title(f'{years[i]} Linear Coefficient')
        ax[1].set_title(f'{years[i]} Residuals')
    return ax
