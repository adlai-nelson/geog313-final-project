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

# write function to get bbox for polygons

def preprocess_data(bbox, start, end, index_name="RVI", scale=2, interval="1Q", deseason=False):
    """
    This function takes 
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
    stack = stackstac.stack(
        items,
        epsg=32737,
        resolution=10,
        bounds_latlon=aoi,
    )
    
    return stack

def coarsen_image(stack, scale = 2):
    coarsened = stack.coarsen(x=scale, y=scale, boundary='pad').mean()
    return coarsened

def temporal_composite(stack, interval = "1Q"):
    composite = stack.resample(time=interval).mean("time")

    return composite

def calculate_index(stack, index):
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
