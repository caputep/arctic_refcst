import os, sys, getopt
import math
import numpy as np
import xarray as xr
import datetime as dt
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.colors as col
from scipy.ndimage.filters import minimum_filter, maximum_filter
from scipy.ndimage.filters import generic_filter as gf
from geopy.distance import great_circle
from geopy.distance import VincentyDistance
import geopy
import pandas as pd

os.environ['DISPLAY']='reed.atmos.albany.edu:1.0'

from netCDF4 import Dataset

#import colorlib
#import metlib
#import funclist

import warnings
warnings.filterwarnings("ignore")

#==========================================================================================

#Directory where everything will be done in
dirpath = "/cstar/burg/"

try:
   opts, args = getopt.getopt(sys.argv[1:],"hd:i:l:",["input_name=","input_id=","leadtime="])
except getopt.GetoptError:
   print('track_ens.py -d <input_file> -i <cyclone_id> -l <leadtime>')
   sys.exit(2)
for opt, arg in opts:
   if opt == '-h':
      print('track_ens.py -d <input_file> -i <cyclone_id> -l <leadtime>')
      sys.exit()
   elif opt in ("-d", "--input_file"):
      input_name = arg
   elif opt in ("-i", "--input_id"):
      input_id = int(arg)
   elif opt in ("-l", "--lead_time"):
      leadtime = int(arg)

#File and cyclone id to read in from sprenger dataset
#input_name = "tr_198501" #name of monthly file for track data
#input_id = 25028 #cyclone ID as in sprenger dataset

#cfsr or erai
model = "gefsr"

#lead time in days (0,1,2,3,etc.)
#leadtime = 1#int(sys.argv[2])

save_map = 1 #set this to 1 to save output images in this directory

if model == "gefsr":
    filepath = ""
    filetype = "nc"
    #vortpath = dirpath + "avg/" + str_date + ".nc"
    
    var_mslp = "PRES_P1_L101_GLL0"
    var_u = "UGRD_P1_L100_GLL0"
    var_v = "VGRD_P1_L100_GLL0"
    var_t = "TMP_P1_L100_GLL0"
    var_g = "HGT_P1_L100_GLL0"
    var_fcst = "forecast_time0"
    var_ens = "ensemble0"
    var_lat = "lat_0"
    var_lon = "lon_0"
    var_lev = "lv_ISBL0"
    
    timestep = 6
    #nwindow = 10 #eps would be 20 for 0.5 deg
    nwindow = 20 #eps would be 40 for 0.5 deg
    reversed_lat = 1 #1 if latitude goes from N to S
    longitude = 360 #180 or 360 degrees
    avg_radius = 7
    vort_thres = 1.5
    
    output_path = "./"
    
if model == "cfsr":
    var_pres = "pmsl"
    var_fcst = "time"
    var_ens = "ensembles"
    var_lat = "lat"
    var_lon = "lon"
    
    timestep = 6
    #nwindow = 10 #eps would be 20 for 0.5 deg
    nwindow = 20 #eps would be 40 for 0.5 deg
    reversed_lat = 0
    avg_radius = 7
    vort_thres = 1.5
    
if model == "erai":
    var_pres = "pmsl"
    var_fcst = "time"
    var_ens = "ensembles"
    var_lat = "lat"
    var_lon = "lon"
    
    timestep = 6
    #nwindow = 10 #eps would be 20 for 0.5 deg
    nwindow = 20 #eps would be 40 for 0.5 deg
    reversed_lat = 0

# does not use these? - has domain in kilometers later

#OUTER DOMAIN: bounds to search cyclones within (specify lons from 0 to 360 degrees)
rbound_s = 60.0 #20.0
rbound_n = 90.0 #75.0
rbound_w = -180.0 #-140.0
rbound_e = 180.0 #-10.0

#INNER DOMAIN: only retain cyclones that pass within this domain
ibound_s = 60.0 #35.0
ibound_n = 90.0 #45.0
ibound_w = -180.0 #-80.0
ibound_e = 180.0 #-60.0

domain = 6.0
if leadtime == 2: domain = 7.0
if leadtime == 3: domain = 7.5
if leadtime == 4: domain = 8.0
if leadtime == 5: domain = 8.5
    
#==========================================================================================
# Supplementary Functions
#==========================================================================================

def calculate_initial_compass_bearing(pointA, pointB):
    import math
    """
    This function is from Github - calculates bearing between two points.
    Parameters:
    - PointA: (lat,lon) for first point
    - PointB: (lat,lon) for second point
    Returns:
    - Bearing in degrees (0-360)
    """
    
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(float(pointB[1]) - float(pointA[1]))

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

#Calculates the average of two angles
def avg_angle(a,b):
    a = a % 360
    b = b % 360
    varsum = a + b
    if varsum > 360 and varsum < 540:
        varsum = varsum % 180
    return varsum / 2

#Supplementary function for the difference between two angles
def new_mod(a,b):
    
    return (a % b + b) % b

#Calculates the difference between two angles
def angle_diff(a,b):
    
    angle = b - a
    angle = new_mod(angle + 180,360) - 180
    
    return angle

#Gaussean smooth
import scipy.ndimage as ndimage
def smooth(prod,sig):
    
    try:
        lats = prod.lat.values
        lons = prod.lon.values
        prod = ndimage.gaussian_filter(prod,sigma=sig,order=0)
        prod = xr.DataArray(prod, coords=[lats, lons], dims=['lat', 'lon'])
    except:
        prod = ndimage.gaussian_filter(prod,sigma=sig,order=0)
    
    return prod

#Find nearest value in a numpy array, return its index within that array
def findNearest(array,val):
    return np.abs(array - val).argmin()

def get_leadtimes(storm_date,peak_time):
    
    #Determine which run to start with
    forward = 0
    max_verif = peak_time - dt.timedelta(hours=12)
    if np.datetime64(max_verif) in storm_date:
        latest_possible_run = (peak_time + dt.timedelta(hours=6)).replace(hour=0)
        earliest_possible_run = (peak_time - dt.timedelta(hours=12)).replace(hour=0)
        valid_times = [peak_time + dt.timedelta(hours=i) for i in [-12,-6,0,6]]
    else:
        forward = 1
        latest_possible_run = (peak_time + dt.timedelta(hours=12)).replace(hour=0)
        earliest_possible_run = (peak_time - dt.timedelta(hours=6)).replace(hour=0)
        valid_times = [peak_time + dt.timedelta(hours=i) for i in [-6,0,6,12]]

    #Construct array of valid times
    leadtime_init = []
    leadtime_valid = []
    leadtime_hour = []
    unique_init = []
    for leadtime in range(0,144,6):
        check_run = latest_possible_run + dt.timedelta(hours=0)
        check_date = latest_possible_run + dt.timedelta(hours=leadtime)
        
        while check_date not in valid_times:
            check_date = check_date - dt.timedelta(hours=24)
            check_run = check_run - dt.timedelta(hours=24)
        leadtime_init.append(check_run)
        leadtime_valid.append(check_date)
        leadtime_hour.append(leadtime)
        if check_run not in unique_init: unique_init.append(check_run)
            
    return unique_init, forward

#Read Sprenger track data, and return data for the passed cyclone ID
def read_sprenger(newid):
    
    storm_date = []
    storm_timepos = []
    storm_lat = []
    storm_lon = []
    storm_mslp = []
    storm_pcont = []
    storm_idclust = []
    storm_label = []
    storm_cid = []
    storm_idcont = []
    storm_mindate = []
    
    def strformat(val):
        if val < 10: return "0"+str(val)
        return str(val)
    

    #Open monthly file
    f = open("/cstar/burg/tracks/sprenger_tracks/"+input_name)
    content = f.readlines()

    storm_date = []
    storm_timepos = []
    storm_lat = []
    storm_lon = []
    storm_mslp = []
    storm_pcont = []
    storm_idclust = []
    storm_label = []
    storm_cid = []
    storm_idcont = []
    storm_mindate = []

    inbox = 0
    min_inbox = 9999
    dmin_km = 9e6
    dmin_date = 0

    #Ignore the first 3 lines
    for line in content[4:]:

        current_inbox = 0

        #Break cyclone apart
        if len(line) == 1:

            #Check here!
            check = True

            #Exit if no storm exists
            if len(storm_mslp) > 0:
                    
                #Append to main array
                if check == True and storm_cid[0] == newid:

                    try:
                        storm_date = storm_date[:]
                        storm_timepos = storm_timepos[:]
                        storm_lat = storm_lat[:]
                        storm_lon = storm_lon[:]
                        storm_mslp = storm_mslp[:]
                        storm_pcont = storm_pcont[:]
                        storm_idclust = storm_idclust[:]
                        storm_label = storm_label[:]
                        storm_cid = storm_cid[:]
                        storm_idcont = storm_idcont[:]
                    except:
                        pass
                    
                    storm_mindate = dmin_date

                    f.close()
                    return storm_date,storm_timepos,storm_lat,storm_lon,storm_mslp,storm_pcont,storm_mindate

            #Clear current arrays for next storm
            storm_date = []
            storm_timepos = []
            storm_lat = []
            storm_lon = []
            storm_mslp = []
            storm_pcont = []
            storm_idclust = []
            storm_label = []
            storm_cid = []
            storm_idcont = []

            inbox = 0
            min_inbox = 9999
            dmin_km = 9e6
            dmin_date = 0

            #Move on to next cyclone
            continue


        #Example:
        #  43824.00     44.00    -62.00    978.31    981.50     26.00      1.00  20640.00    192.00
        #012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789
        #0         1         2         3         4         5         6         7         8

        #Split line by variables
        timepos = float(line[0:10])
        lon = float(line[11:20])
        lat = float(line[21:30])
        mslp = float(line[31:40])
        pcont = float(line[41:50])
        idclust = int(float(line[51:60]))
        label = int(float(line[61:70]))
        cid = int(float(line[71:80]))
        idcont = int(float(line[81:90]))

        #Fix lon to be from 0 to 360
        olon = lon
        if lon < 0.0: lon = 360.0 + lon
        if model == 'erai': olon = lon

        #Calculate date
        sdate = dt.datetime(1979,1,1,0)
        cdate = sdate + dt.timedelta(hours=timepos)

        #Check for closest proximity to 40N/70W benchmark
        start = (lat, lon)
        end = (40.0, 290.0)
        dist = great_circle(start, end).kilometers
        if dist < dmin_km:
            dmin_km = dist
            dmin_date = cdate

        #Append to main array
        storm_date.append(np.datetime64(cdate))
        storm_timepos.append(timepos)
        storm_lat.append(lat)
        storm_lon.append(olon)
        storm_mslp.append(mslp)
        storm_pcont.append(pcont)
        storm_idclust.append(idclust)
        storm_label.append(label)
        storm_cid.append(cid)
        storm_idcont.append(idcont)

    #Close file
    f.close()

    return 0

def read_tracks(newid):
    
#    f = open("/cstar/burg/tracks/cfsr_tracks/"+newid+".csv","r")
    f = open("./"+newid+".csv","r")
    lines = f.readlines()
    lineArray = lines[0].split("\n")[0]
    lineArray = lineArray.split(";")
    
    storm_lat = map(float, lineArray[0].split(","))
    storm_lon = map(float, lineArray[1].split(","))
    storm_hght = map(float, lineArray[2].split(","))
    storm_vort = map(float, lineArray[3].split(","))
    
    storm_lat = [float(i) for i in lineArray[0].split(",")]
    storm_lon = [float(i) for i in lineArray[1].split(",")]
    storm_hght = [float(i) for i in lineArray[2].split(",")]
    storm_vort = [float(i) for i in lineArray[3].split(",")]
    storm_time = lineArray[4].split(",")
    storm_time = [np.datetime64(dt.datetime.strptime(i,'%Y%m%d%H')) for i in storm_time]
    f.close()
    
    
    storm_lon = np.array(storm_lon)
    storm_lat = np.array(storm_lat)
    storm_hght = np.array(storm_hght)
    storm_vort = np.array(storm_vort)
    storm_time = np.array(storm_time)

    #Determine time of peak impact
    storm_dist = [great_circle((storm_lat[i],storm_lon[i]), (40.0,-70.0)).kilometers for i in range(len(storm_lon))]
    storm_dist = np.array(storm_dist)
    peak_time_idx = np.argmin(storm_dist)
    peak_time = storm_time[peak_time_idx].astype(dt.datetime)
    
    #Change peak time if necessary
    storm_dates = [pd.to_datetime(j) for j in storm_time]
    if storm_dates.index(peak_time) == 0:
        peak_time = storm_dates[1]
    elif storm_dates.index(peak_time) == 1:
        peak_time = storm_dates[2]
    elif storm_dates.index(peak_time) == len(storm_dates)-1:
        peak_time = storm_dates[-2]
    
    return storm_time, storm_lat, storm_lon, storm_hght, storm_vort, peak_time

#radius = rad in km for subsetting data by
def calc_centroid(center_lon, center_lat, radius, calc_type, mslp, lats, lons):
    """
    Calculates a centroid given an initial latitude/longitude and a 2D array.
    
    Parameters:
    ----------------
    center_lon : float
        Float representing the original center longitude, in degrees
    center_lat : float
        Float representing the original center latitude, in degrees
    radius : float
        Radius over which to calculate the centroid, in kilometers
    calc_type : string
        Type of centroid to perform, can be "max" or "min"
    mslp : array
        2D array containing the data (e.g., MSLP, vorticity) for calculating the
        centroid
    lats : array
        1D array containing latitudes corresponding to "mslp"
    lons : array
        1D array containing longitudes corresponding to "mslp"
        
    Returns:
    -----------------
    lon_final : float
        Longitude of the centroid
    lat_final : float
        Latitude of the centroid
    val_final : float
        Absolute minimum MSLP value within the radius (not at the lon_final & lat_final point)
    """
    reversed_lat = 0
    
    #----------------------------------------------------------------------
    # 1. Start out with the absolute min within this radius
    #----------------------------------------------------------------------
    
    #Mask out all values outside of the requested radius
    bound_mask = np.zeros(mslp.values.shape)
    mslp_vals = mslp.values
    for j in range(len(lats)):
        for i in range(len(lons)):
            ilat = lats[j]
            ilon = lons[i]
            if great_circle((center_lat,center_lon),(ilat,ilon)).kilometers <= (radius):
                bound_mask[j][i] = 1
    subset_var = mslp_vals + 0.0
    subset_var[bound_mask==0] = np.nan

    #Identify the lat & lon of the absolute min/max within the radius
    if calc_type == "min": val_final = np.nanmin(subset_var) + 0.0
    if calc_type == "max": val_final = np.nanmax(subset_var) + 0.0
    lon_final = lons[np.where(subset_var == val_final)[1][0]]
    lat_final = lats[np.where(subset_var == val_final)[0][0]]
    val_lat = lat_final
    val_lon = lon_final
    
    #----------------------------------------------------------------------
    # 2. Set up for calculation
    #----------------------------------------------------------------------

    #Coordinates of previous lat & lon in iteration
    prev_lat = 0.0
    prev_lon = 0.0

    #Coordinates of current lat & lon in iteration
    this_lat = val_lat
    this_lon = val_lon
    
    #Calculate environmental value of variable at the requested radius
    bound_lats = []
    bound_lons = []
    bound_pairs = []
    for k in np.arange(0,361):
        origin = geopy.Point(this_lat,this_lon) #first
        destination = great_circle(kilometers=radius).destination(origin, k)
        proj_lat, proj_lon = destination.latitude, destination.longitude
        flat = lats[findNearest(lats,proj_lat)]
        flon = lons[findNearest(lons,proj_lon)]
        if (flat,flon) not in bound_pairs:
            bound_pairs.append((flat,flon))
            bound_lats.append(flat)
            bound_lons.append(flon)

    #Environmental value of variable
    env_g = []
    temp_lats = mslp.lat.values
    temp_lons = mslp.lon.values
    for k in bound_pairs:
        latidx = np.where(temp_lats==k[0])[0][0]
        lonidx = np.where(temp_lons==k[1])[0][0]
        env_g.append(mslp_vals[latidx][lonidx])
    if calc_type == "min": env_val = np.max(env_g)
    if calc_type == "max": env_val = np.min(env_g)
    
    #----------------------------------------------------------------------
    # 3. Calculate the centroid
    #----------------------------------------------------------------------

    #Stop if too many loops!
    loopcheck = 0
    
    #Loop while the difference between the old and new position is non-negligible
    while ( math.sqrt((this_lat - prev_lat)**2 + (this_lon - prev_lon)**2) > 0.12 ):
        
        #Break if too many loops
        loopcheck += 1
        if loopcheck >= 10: break

        #Set previous lat & lon to current values for next iteration
        prev_lat = this_lat
        prev_lon = this_lon
        
        #------------------------------------------------------------------
        # 3a. Subset array around cyclone & get environmental value
        #------------------------------------------------------------------
        
        #Break if latitude is outside of range
        if this_lat > 90 or this_lat < 0:
            return 0.0, 0.0, 0.0
        
        #Recalculate environmental value
        bound_lats = []
        bound_lons = []
        bound_pairs = []

        for k in np.arange(0,361):
            origin = geopy.Point(this_lat,this_lon) #second
            destination = great_circle(kilometers=radius).destination(origin, k)
            proj_lat, proj_lon = destination.latitude, destination.longitude
            flat = lats[findNearest(lats,proj_lat)]
            flon = lons[findNearest(lons,proj_lon)]
            if (flat,flon) not in bound_pairs:
                bound_pairs.append((flat,flon))
                bound_lats.append(flat)
                bound_lons.append(flon)

        #Environmental value of variable
        env_g = []
        for k in bound_pairs:
            latidx = np.where(temp_lats==k[0])[0][0]
            lonidx = np.where(temp_lons==k[1])[0][0]
            env_g.append(mslp_vals[latidx][lonidx])
        if calc_type == "min": env_val = np.max(env_g)
        if calc_type == "max": env_val = np.min(env_g)
        
        #Subset array surrounding first guess and mask values outside of specified radius
        subset_var_nomask = mslp.sel(lat=slice(min(bound_lats),max(bound_lats)),lon=slice(min(bound_lons),max(bound_lons)))
        subset_lats = subset_var_nomask.lat.values
        subset_lons = subset_var_nomask.lon.values
        subset_var_nomask_vals = subset_var_nomask.values
        
        bound_mask = np.zeros(subset_var_nomask_vals.shape)
        for j in range(len(subset_lats)):
            for i in range(len(subset_lons)):
                ilat = subset_lats[j]
                ilon = subset_lons[i]
                if great_circle((this_lat,this_lon),(ilat,ilon)).kilometers <= radius:
                    bound_mask[j][i] = 1

        subset_var = subset_var_nomask_vals + 0.0
        subset_var[bound_mask==0] = np.nan #MONKEY
        if calc_type == "min": val_final = np.nanmin(subset_var)
        if calc_type == "max": val_final = np.nanmax(subset_var)
        
        #Break if there is an error, meaning cyclone is outside of bounds
        if len(subset_lats) < 4 or len(subset_lons) < 4:
            return 0.0, 0.0, 0.0


        #------------------------------------------------------------------
        # 3b. compute the integral
        #------------------------------------------------------------------
        
        #Get Pi for gridbox (Pi = Penv - Pval)
        diff = env_val - subset_var
        
        #Get current lat and lon values
        slons2d,slats2d = np.meshgrid(subset_lons,subset_lats)
        
        #final lat = sum(lat * Pi) / sum(Pi)
        #latval = sum(lat * Pi)
        #diffsum = sum(Pi)
        latval = np.nansum(slats2d * diff)
        lonval = np.nansum(slons2d * diff)
        diffsum = np.nansum(diff)
        
        #-----------------------------------------------------------

        #Average it all out
        val_lat = latval / diffsum
        val_lon = lonval / diffsum

        #Save the result for the next iteration to determine whether to stop looping
        this_lat = float(val_lat)
        this_lon = float(val_lon)

        #If this is last iteration, save the final position for the output file
        lon_final = round(val_lon,1)
        lat_final = round(val_lat,1)

        val_lat = this_lat
        val_lon = this_lon
        
        #Round to 3 nearest decimal points
        this_lat = round(this_lat,4)
        this_lon = round(this_lon,4)
        prev_lat = round(prev_lat,4)
        prev_lon = round(prev_lon,4)

    
    #Error check: If val is nan
    if math.isnan(lat_final) == True or math.isnan(lon_final) == True:
        #print("--> centroid algorithm failed, nan value found")
        return 0.0, 0.0, 0.0
    
    #Error check: is there a larger min/max just outside this radius? Check min/max values
    #within 300 and 350 km radii away from the centroid. If 350km has a larger min/max than 
    #300km, it's not a proper radius.
    mins = []
    mins_dist = []
    mins_lats = []
    mslp_vals = mslp.values
    for irad in [300,350]: #radius+i, and 50,150
        bound_mask = np.zeros(mslp_vals.shape)
        for j in range(len(lats)):
            for i in range(len(lons)):
                ilat = lats[j]
                ilon = lons[i]
                if great_circle((lat_final,lon_final),(ilat,ilon)).kilometers <= (irad):
                    bound_mask[j][i] = 1
        #subset_var = np.ma.masked_where(bound_mask==0,subset_var.values)
        subset_var = mslp_vals + 0.0
        subset_var[bound_mask==0] = np.nan
        
        lonsss = lons[np.where(mslp_vals == np.nanmin(subset_var))[1][0]]
        latsss = lats[np.where(mslp_vals == np.nanmin(subset_var))[0][0]]
        if calc_type == "min": mins.append(np.nanmin(subset_var))
        if calc_type == "max": mins.append(np.nanmax(subset_var))
        mins_dist.append(great_circle((lat_final,lon_final),(latsss,lonsss)).kilometers)
    
    check_fail = mins[1] < mins[0]
    shift_dist = great_circle((center_lat,center_lon),(lat_final,lon_final)).kilometers
    env_diff = abs(val_final - env_val)
    
    if calc_type == "min": check_fail = mins[1] < (mins[0]-100) #allow for 100m difference in height 
    if calc_type == "max": check_fail = mins[1] > (mins[0]+100) #allow for 100m difference in height
    if mins[1] < (mins[0]-50):
        return 0.0, 0.0, 0.0
    
    #Error check: If distance between start & end point is bigger than the radius
    if shift_dist >= 700.0:
        check2 = env_diff > 100.0 and shift_dist <= 750.0
        if check2 == False:
            return 0.0, 0.0, 0.0

    return lon_final, lat_final, val_final

import os, sys
import numpy as np
import xarray as xr
from metpy.units import units
from metpy.xarray import preprocess_xarray
import metpy.calc as calc

@preprocess_xarray
def calcavg(x,xavg,lon2d,lat2d,nlon,nlat,rad,box,eqrm):
    
    nbox = (2*box+1)*(2*box+1)
    
    #Iterate over latitude and longitude
    for j in range((box),(nlon-box)):
        for i in range((box),(nlat-box)):

            lon1d = lon2d[i-box:i+box+1,j-box:j+box+1].reshape((nbox))
            lat1d = lat2d[i-box:i+box+1,j-box:j+box+1].reshape((nbox))
            x1d = x[i-box:i+box+1,j-box:j+box+1].reshape((nbox))

            d1d = eqrm * np.sqrt(( (lon2d[i,j]-lon1d)*np.cos( (lat2d[i,j]+lat1d)/2.0 ) )**2.0 + (lat2d[i,j]-lat1d)**2.0)
            z = x1d[d1d < rad] / len(x1d[d1d < rad])
            xavg[i,j] = np.sum(z)
   
    return xavg

@preprocess_xarray
def area_average(var,rad,lon,lat):
    
    """Performs horizontal area-averaging of a field in latitude/longitude format.
    Parameters
    ----------
    var : (M, N) ndarray
        Variable to perform area averaging on. Can be 2, 3 or 4 dimensions. If 2D, coordinates must
        be lat/lon. If using additional dimensions, area-averaging will only be performed on the last
        2 dimensions, assuming those are latitude and longitude.
    rad : `pint.Quantity`
        The radius over which to perform the spatial area-averaging.
    lon : array-like
        Array of longitudes defining the grid
    lat : array-like
        Array of latitudes defining the grid
        
    Returns
    -------
    (M, N) ndarray
        Area-averaged quantity, returned in the same dimensions as passed.
    
    Notes
    -----
    This function was originally provided by Matthew Janiga and Philippe Papin using a Fortran wrapper for NCL,
    and converted to python with further efficiency modifications by Tomer Burg, with permission from the original
    authors.
    
    This function assumes that the last 2 dimensions of var are ordered as (....,lat,lon).
    """
    
    #convert radius to kilometers
    rad = rad.to('kilometers')
    
    #res = distance in km of dataset resolution, at the equator
    londiff = lon[1]-lon[0]
    latdiff = lat[1]-lat[0]
    lat_0 = 0.0 - (latdiff/2.0)
    lat_1 = 0.0 + (latdiff/2.0)
    dx,dy = calc.lat_lon_grid_deltas(np.array([lon[0],lon[1]]), np.array([lat_0,lat_1]))
    dx = dx.to('km')
    res = int((dx[0].magnitude + dx[1].magnitude)/2.0) * units('km')
    
    #---------------------------------------------------------------------
    #Error checks
    
    #Check to make sure latitudes increase
    reversed_lat = 0
    if lat[1] < lat[0]:
        reversed_lat = 1
        
        #Reverse latitude array
        lat = lat[::-1]
        
        #Determine which axis of variable array to reverse
        lat_dim = len(var.shape)-2
        var = np.flip(var,lat_dim)
        
    #Check to ensure input array has 2, 3 or 4 dimensions
    var_dims = np.shape(var)
    if len(var_dims) not in [2,3,4]:
        print("only 2D, 3D and 4D arrays allowed")
        return
    
    #---------------------------------------------------------------------
    #Prepare for computation
    
    #Number of points in circle (with buffer)
    box = int((rad/res)+2)

    #Define empty average array
    var_avg = np.zeros((var.shape))
        
    #Convert lat and lon arrays to 2D
    nlat = len(lat)
    nlon = len(lon)
    lon2d,lat2d = np.meshgrid(lon,lat)
    RPD = 0.0174532925
    lat2d = lat2d*RPD
    lon2d = lon2d*RPD

    #Define radius of earth in km
    eqrm = 6378.137
    
    #Create mask for elements of array that are outside of the box
    mask = np.zeros((lon2d.shape))
    nbox = (2*box+1)*(2*box+1)
    mask[box:nlat-box,box:nlon-box] = 1
    mask[mask==0] = np.nan
    
    #Calculate area-averaging depending on the dimension sizes
    if len(var_dims) == 2:
        var_avg = calcavg(var.magnitude, var_avg, lon2d, lat2d, nlon, nlat, rad.magnitude, box, eqrm) * mask
    elif len(var_dims) == 3:
        for t in range(var_dims[0]):
            var_avg[t,:,:] = calcavg(var[t,:,:].magnitude, var_avg[t,:,:], lon2d, lat2d, nlon, nlat, rad.magnitude, box, eqrm) * mask
    elif len(var_dims) == 4:
        for t in range(var_dims[0]):
            for l in range(var_dims[1]):
                var_avg[t,l,:,:] = calcavg(var[t,l,:,:].magnitude, var_avg[t,l,:,:], lon2d, lat2d, nlon, nlat, rad.magnitude, box, eqrm) * mask
                
    #If latitude is reversed, then flip it back to its original order
    if reversed_lat == 1:
        lat_dim = len(var.shape)-2
        var_avg = np.flip(var_avg,lat_dim)
    
    #Return area-averaged array with the same units as the input variable
    return var_avg * var.units

#==========================================================================================
# Set up tracking algorithm
#==========================================================================================

#Step 0: get data about case
#storm_date,storm_lat,storm_lon,storm_hght,storm_vort,peak_time = read_tracks(input_id)
storm_date,storm_timepos,storm_lat,storm_lon,storm_mslp,storm_env,peak_time = read_sprenger(input_id)

#Convert storm dates from np.datetime64 to datetime (to make use of datetime's date formatting functions)
storm_dates = [pd.to_datetime(i) for i in storm_date]

#Step 1: Check the dates that the cyclone exists during
start_date = storm_date[0].astype(dt.datetime)
end_date = storm_date[-1].astype(dt.datetime)

#Set however many lead time days were passed into the script
case_runs, forward = get_leadtimes(storm_date,peak_time)
init_date = case_runs[leadtime]

### CHANGE HERE ###
# This is where you determine what file to open. The file should contain a single initialization date,
# multiple ensembles and forecast hours.
#data = xr.open_dataset("/cstar/burg/data/gefs925/"+dt.datetime.strftime(init_date,"%Y%m%d")+".nc")
data = xr.open_dataset("./reforecast_19870802.nc")

#=========================================================================================
# Part 0b: Subset the data to the times the cyclone was active during
#=========================================================================================

#Subset the data by time from cyclogenesis to end of cyclogenesis

#Step 1. Convert time steps to forecast hours
exec("data_fhr = data."+var_fcst+".values")
data_fhr = data_fhr / np.timedelta64(1, 'D')
data_fhr = data_fhr * 24.0

#Step 2. convert forecast hours to valid time steps
data_valid = [init_date + dt.timedelta(hours=i) for i in data_fhr]

#Step 3. Get the start & end dates to track
start_date = storm_date[0].astype(dt.datetime)
end_date = storm_date[-1].astype(dt.datetime)
gefs_sdate = start_date - dt.timedelta(hours=0)
gefs_edate = end_date + dt.timedelta(hours=12)

if gefs_sdate not in data_valid: gefs_sdate = data_valid[0]
if gefs_edate not in data_valid: gefs_edate = data_valid[-1]
idx_start = data_valid.index(gefs_sdate)
idx_end = data_valid.index(gefs_edate)

#just rearranging some stuff
data_full = data.rename({var_lon:"lon",var_lat:"lat",var_fcst:"time",var_lev:"lev"})
newlevs = data_full.lev.values / 100.0
data_full['lev'] = newlevs

#Step 4. Subset data by storm valid times
data_all = data_full.isel(time=slice(idx_start,idx_end))
data_fhr = data_all.time.values
data_fhr = data_fhr / np.timedelta64(1, 'D')
data_fhr = data_fhr * 24.0
data_valid = [init_date + dt.timedelta(hours=i) for i in data_fhr]

lats = data_full.lat.values
lons = data_full.lon.values
lons2d, lats2d = np.meshgrid(lons,lats)

#=========================================================================================================
#Step 2: Load data & identify minima in E Coast
#=========================================================================================================

try:
    m
except:
    mbound_n = 90.0 #57.0
    mbound_s = 60.0 #7.0
    mbound_w = 0 #-138.0 + 360.0
    mbound_e = 360 #-15.0 + 360.0
    
    mbound_s = 60.0 #4.0
    mbound_n = 90.0 #70.0
    mbound_w = 0 #-120.0+360.0
    mbound_e = 360 #0.0+360.0
    lat_1 = 0.0
    lat_2 = 90.0 #57.0
    lat_0 = 60.0 #10.0
    lon_0 = 0.0 #-70.0+360.0
    latfac = 20.0
    lonfac = 40.0
    #nres = 'i'
    
    #m = Basemap(lat_1=lat_1,lat_2=lat_2,lat_0=lat_0,lon_0=lon_0,llcrnrlat=mbound_s,urcrnrlat=mbound_n,llcrnrlon=mbound_w,urcrnrlon=mbound_e,rsphere=(6378137.00,6356752.3142),resolution=nres,area_thresh=1000.0,projection='lcc')

    #m = Basemap(lat_1=lat_1,lat_2=lat_2,lat_0=lat_0,lon_0=lon_0,crnrlat=mbound_s,urcrnrlat=mbound_n,llcrnrlon=mbound_w,urcrnrlon=mbound_e,resolution='i',area_thresh=1000.0,projection='npaeqd')

    m = Basemap(projection='npaeqd',boundinglat=60,lon_0=270,resolution='i')

saveline = ""
for ens in data_all.ensemble0.values:
    data_ens = data_all.sel(ensemble0=ens)
    
    #Subset the dataset by time
#    data_mslp = np.multiply(data_ens[var_mslp],0.01)
    data_u = data_ens[var_u]
    data_v = data_ens[var_v]
    data_g = data_ens[var_g]
    #data_vort = data_ens['vort_avg']
    
    #Flip latitudes and create full dataset
    #---------- CHANGE HERE -----------------------------------
    #If latitude is from north to south, run this line:
    data_full2 = xr.merge([data_u[:,:,::-1,:],data_v[:,:,::-1,:],data_g[:,:,::-1,:]])
    
    #If latitude is from south to north, run this line:
    #data_full2 = xr.merge([data_u,data_v,data_g])
    #----------------------------------------------------------
    
    reversed_lat = 0
    data_full2['lon'] = data_full2['lon'].values - 360.0
    #print(data_full.lon.values)
    data_full2 = data_full2.rename({var_v:"v",var_u:"u",var_g:"g"})
    data_full = data_full2.sel(lat=slice(rbound_s,rbound_n),lon=slice(rbound_w,rbound_e))
    lats = data_full.lat.values
    lons = data_full.lon.values
    lons2d, lats2d = np.meshgrid(lons,lats)
    
    #Step 3: Initialize the domain near the cyclogenesis point (keeping in mind cfsr is -180 to 180, -90 to 90)
    start_lat = storm_lat[0]
    start_lon = storm_lon[0]

    #domain = 6.0 #Use large domain at first

    #Initial domain to search for cyclone within
    abound_n = start_lat + domain
    abound_s = start_lat - domain
    abound_w = start_lon - domain
    abound_e = start_lon + domain
    if reversed_lat == 1:
        abound_n = start_lat - domain
        abound_s = start_lat + domain
    if longitude == 360:
        abound_w += 360.0
        abound_e += 360.0

    #List containing dicts about every storm
    storms = []
    cid = 0

    #Loop through all time steps and search for minimums
    time_vals = data_full.time.values
    for i_hr in range(len(time_vals)):

        #=====================================================================================
        # 0. subset data by time and calculate area-averaged relative vorticity
        #=====================================================================================

        #Subset by time
        data = data_full.isel(time=i_hr)
        valid_obj = data_valid[i_hr]
        valid_str = dt.datetime.strftime(valid_obj,'%Y%m%d %H00')
        #print(f"--> tracking cyclones for {valid_str}")

        #Retrieve area-averaged vorticity with a 2 gaussean filter
        #data_full = data_full2.sel(lat=slice(rbound_s,rbound_n),lon=slice(rbound_w,rbound_e))
        u_full = data_full2.u.isel(time=i_hr)
        v_full = data_full2.v.isel(time=i_hr)
        lons_full = data_full2.lon.values
        lats_full = data_full2.lat.values
        radius = 700.0 * units('kilometers')
        dx, dy = calc.lat_lon_grid_deltas(lons_full,lats_full) #get lat/lon grid deltas
        vort = calc.vorticity(u_full.sel(lev=925)*units('m/s'),v_full.sel(lev=925)*units('m/s'),dx,dy) #calculate vorticity using metpy
        data_vort = area_average(vort,radius,lons_full+360.0,lats_full) #calculate area-averaged vorticity
        data_vort = xr.DataArray(data_vort,coords=[lats_full,lons_full],dims=['lat','lon']) #convert to dataarray
        data_vort = data_vort.sel(lat=slice(rbound_s,rbound_n),lon=slice(rbound_w,rbound_e))
        data['vort_avg'] = data_vort #append to full dataset
        data_vort = smooth(data_vort,2).values * 10**5   #metlib.smooth(data_vort,2).values * 10**5
        data_vort = np.nan_to_num(data_vort) #temporary fix for nan values
        

        #=====================================================================================
        # 1. find all vorticity maxima in field & apply height centroid
        #=====================================================================================

        #Search for all maxima in the vorticity field
        res = lats[1] - lats[0]
        nwindow = int(10.0 / res) #every 20 degrees? then 10
        max_all = maximum_filter(data_vort, size=nwindow) #20
        max_all = np.nonzero(data_vort == max_all)
        max_all_lons = lons2d[max_all];
        max_all_lats = lats2d[max_all];
        max_all_vals = data_vort[max_all];

        #Preliminary list of cyclones #0
        imax_lons_0 = []
        imax_lats_0 = []
        imax_vals_0 = []
        imax_found_0 = []
        imax_pairs_0 = []

        #Loop through all minima found, and remove minima outside of this domain
        for x in range(0,len(max_all_vals)):

            #Only retain minima in lat/lon subset region
            if max_all_lons[x] in lons[5:-5] and max_all_lats[x] in lats[5:-5]:

                #Don't append the same cyclone twice!
                #if max_all_lons[x] not in imax_lons_0 and max_all_lats[x] not in imax_lats_0:
                if (max_all_lats[x],max_all_lons[x]) not in imax_pairs_0:

                    #Ignore MSLP over 1016 mb
                    vort_thres_min = vort_thres - 0.5 #5.5 for 850mb, 4.0 for 950mb
                    #with 19, this was 4.0, 26 = 3.0, 32/34 = 2.2
                    if max_all_vals[x] >= vort_thres_min:
                        imax_lons_0.append(max_all_lons[x])
                        imax_lats_0.append(max_all_lats[x])
                        imax_vals_0.append(max_all_vals[x])
                        imax_found_0.append(0)
                        imax_pairs_0.append((max_all_lats[x],max_all_lons[x]))

        #For each vort max, find the average height in the 500km radius
        imax_lons = []
        imax_lats = []
        imax_vort = []
        imax_hght = []
        imax_hghtlats = []
        imax_hghtlons = []
        imax_found = []
        data_vort_xr = xr.DataArray(data_vort,coords=[lats,lons],dims=['lat','lon'])
        for j in np.arange(len(imax_lons_0)): #used to be 6.0 radius

            #Calculate centroid using a 500km radius
            #imax_lons_0[j], imax_lats_0[j], new_vort = calc_centroid(imax_lons_0[j], imax_lats_0[j], 500, 'max', data.vort_avg, lats, lons)
            new_mslplon, new_mslplat, new_hght = calc_centroid(imax_lons_0[j], imax_lats_0[j], 500, 'min', data.g.sel(lev=925), lats, lons)

            #If the centroid returned an error, then disregard this vort max
            if new_mslplat < 5: continue

            #Append to list of cyclones #1
            imax_lons.append(imax_lons_0[j])
            imax_lats.append(imax_lats_0[j])
            imax_vort.append(imax_vals_0[j])
            imax_hght.append(new_hght*0.1)
            imax_hghtlats.append(new_mslplat)
            imax_hghtlons.append(new_mslplon)
            imax_found.append(0)


        #=====================================================================================
        # 2. filter out double cyclones & if there's several cyclones w/in close proximiy, keep the deeper one
        #=====================================================================================

        if len(imax_lons) > 1:

            #Check if there's multiple cyclones within close proximity
            max_lons = []
            max_lats = []
            max_vort = []
            max_hght = []
            max_hghtlats = []
            max_hghtlons = []
            max_found = []
            blacklist = []

            #Loop through all cyclones
            for x in range(len(imax_lons)):

                if len(blacklist) > 0 and x in blacklist: continue
                appended = 0

                #Loop through all cyclones
                for y in range(len(imax_lons)):

                    if x == y: continue

                    if len(blacklist) > 0 and y in blacklist: continue

                    #Get distance between current cyclone and searched cyclone
                    start_loc1 = (imax_lats[x],imax_lons[x])
                    end_loc1 = (imax_lats[y],imax_lons[y])
                    dist_vort = great_circle(start_loc1,end_loc1).kilometers

                    start_loc2 = (imax_hghtlats[x],imax_hghtlons[x])
                    end_loc2 = (imax_hghtlats[y],imax_hghtlons[y])
                    dist_hght = great_circle(start_loc2,end_loc2).kilometers

                    dist_between_x = great_circle(start_loc1,start_loc2).kilometers
                    dist_between_y = great_circle(end_loc1,end_loc2).kilometers

                    #Compare vort max with height min of opposite storm
                    dist_between_xy = great_circle(start_loc1,end_loc2).kilometers #vort max of x with height min of y
                    dist_between_yx = great_circle(start_loc2,end_loc1).kilometers
                    check_dist1 = dist_between_xy < max(dist_between_x,dist_between_y)
                    check_dist2 = dist_between_yx < max(dist_between_x,dist_between_y)
                    check_dist = check_dist1 == True and check_dist2 == True

                    check_dist = check_dist == True or dist_hght < min(dist_between_x,dist_between_y)
                    if dist_hght < 250: check_dist = True

                    #If distance is small, then retain stronger one
                    if check_dist == True and (dist_vort < 750 or dist_hght < 750): #used to be 1000, then 1500

                        #Choose stronger one
                        val1 = imax_vort[x]
                        val2 = imax_vort[y]

                        if val1 > val2:
                            max_lons.append(imax_lons[x])
                            max_lats.append(imax_lats[x])
                            max_vort.append(imax_vort[x])
                            max_hght.append(imax_hght[x])
                            max_hghtlats.append(imax_hghtlats[x])
                            max_hghtlons.append(imax_hghtlons[x])
                            max_found.append(0)
                            blacklist.append(y)
                            appended = 1
                        else:
                            max_lons.append(imax_lons[y])
                            max_lats.append(imax_lats[y])
                            max_vort.append(imax_vort[y])
                            max_hght.append(imax_hght[y])
                            max_hghtlats.append(imax_hghtlats[y])
                            max_hghtlons.append(imax_hghtlons[y])
                            max_found.append(0)
                            blacklist.append(x)
                            appended = 1

                if appended == 0:
                    max_lons.append(imax_lons[x])
                    max_lats.append(imax_lats[x])
                    max_vort.append(imax_vort[x])
                    max_hght.append(imax_hght[x])
                    max_hghtlats.append(imax_hghtlats[x])
                    max_hghtlons.append(imax_hghtlons[x])
                    max_found.append(0)

        else:

            max_lons = imax_lons
            max_lats = imax_lats
            max_vort = imax_vort
            max_hght = imax_hght
            max_hghtlats = imax_hghtlats
            max_hghtlons = imax_hghtlons
            max_found = imax_found

        #"""

        #Remove duplicates
        new_maxlons = []
        new_maxlats = []
        new_maxvort = []
        new_maxhght = []
        new_maxhghtlats = []
        new_maxhghtlons = []
        new_maxfound = []
        new_maxpairs = []
        for j in np.arange(len(max_lons)):
            ilat = max_lats[j]
            ilon = max_lons[j]
            #if ilat not in new_maxlats and ilon not in new_maxlons:
            if (ilat,ilon) not in new_maxpairs:
                if np.isnan(ilat) == False:
                    new_maxlats.append(ilat)
                    new_maxlons.append(ilon)
                    new_maxvort.append(max_vort[j])
                    new_maxhght.append(max_hght[j])
                    new_maxhghtlats.append(max_hghtlats[j])
                    new_maxhghtlons.append(max_hghtlons[j])
                    new_maxfound.append(0)
                    new_maxpairs.append((ilat,ilon))
        max_lats = new_maxlats
        max_lons = new_maxlons
        max_vort = new_maxvort
        max_hght = new_maxhght
        max_hghtlats = new_maxhghtlats
        max_hghtlons = new_maxhghtlons
        max_found = new_maxfound
        #"""

        #=====================================================================================
        # 3. append cyclone to total database & forward-track existing cyclones
        #=====================================================================================

        #Skip if no cyclones were identified at this time step
        if len(max_vort) == 0: continue

        #If there are no existing cyclones in the database, consider them all as new cyclones
        if len(storms) == 0:

            #Append cyclones
            for j in np.arange(len(max_vort)):
                if max_vort[j] > vort_thres:
                    storms.append({"id":cid,"vort_lats":[max_lats[j]],"vort_lons":[max_lons[j]],"vort_max":[max_vort[j]],"hght_lats":[max_hghtlats[j]],"hght_lons":[max_hghtlons[j]],"hght_min":[max_hght[j]],"time":[valid_obj]})
                    cid += 1

                #"id","vort_lats","vort_lons","vort_max","hght_lats","hght_lons","hght_min","time"

        #Otherwise, check if cyclone is new or belongs to a previous cyclone
        else:

            #---------------------------------------------------------------------------------
            # a. Loop through all currently identified cyclones and match them by motion vector
            #---------------------------------------------------------------------------------

            #///////////////////// MAP CODE /////////////////////
            if save_map == 1:
                fig,ax=plt.subplots(figsize=(15,12),dpi=125)
                m.drawstates()
                m.drawcountries()
                m.drawcoastlines()
            #///////////////////// MAP CODE /////////////////////

            #Determine previously existing IDs that were already used
            used_id = []

            #Loop through all cyclones currently identified
            for j in np.arange(len(max_vort)):

                max_disthght = []
                max_distvort = []
                max_id = []

                #Loop through all storms that previously exist
                for k in np.arange(len(storms)):

                    #Ignore if this storm is already matched
                    if k in used_id: continue

                    #Only retain storms that existed 6 hours ago
                    diff1 = (valid_obj - storms[k]['time'][-1]).seconds / 3600.0
                    diff2 = (valid_obj - storms[k]['time'][-1]).days * 24.0
                    diff = diff1 + diff2
                    if diff != 6: continue

                    #Check for closest 850-hPa height min that exists within 750km radius
                    last_lat = storms[k]['hght_lats'][-1]
                    last_lon = storms[k]['hght_lons'][-1]
                    last_latvort = storms[k]['vort_lats'][-1]
                    last_lonvort = storms[k]['vort_lons'][-1]

                    #Check to see if new vort max is within similar location
                    #max_dist = np.array([great_circle((last_lat,last_lon),(max_lats[ij],max_lons[ij])).kilometers for ij in np.arange(len(max_lats))])
                    max_disthght.append(great_circle((last_lat,last_lon),(max_hghtlats[j],max_hghtlons[j])).kilometers)
                    max_distvort.append(great_circle((last_latvort,last_lonvort),(max_lats[j],max_lons[j])).kilometers)
                    max_id.append(k)

                #Sort the distance from shortest to largest
                max_disthght = np.array(max_disthght)
                max_id = np.array(max_id)
                max_distvort = np.array(max_distvort)

                sort_args = np.argsort(max_disthght)
                sort_disthght = (max_disthght[sort_args])
                sort_distvort = (max_distvort[sort_args])
                sort_id_full = (max_id[sort_args])

                #Weighted vort & height sorting
                weight_sort_args = np.argsort(max_disthght * max_distvort)
                weight_sort_disthght = (max_disthght[sort_args])
                weight_sort_distvort = (max_distvort[sort_args])
                weight_sort_id_full = (max_id[sort_args])

                #If <=600km for BOTH height and vorticity, add the smallest distance to this cyclone
                found = 0
                if len(sort_disthght) > 0 and len(sort_distvort) > 0:

                    if sort_disthght[0] <= 600.0 and sort_distvort[0] <= 600.0:

                        #Loop through all IDs
                        sort_id = sort_id_full[sort_disthght <= 600.0]

                        #Append by distance if the previous cyclone only has 1 time step
                        len_track = len(storms[sort_id[0]]['hght_lats'])
                        if len_track == 1:
                            storms[sort_id[0]]['vort_lats'].append(max_lats[j])
                            storms[sort_id[0]]['vort_lons'].append(max_lons[j])
                            storms[sort_id[0]]['vort_max'].append(max_vort[j])
                            storms[sort_id[0]]['hght_lats'].append(max_hghtlats[j])
                            storms[sort_id[0]]['hght_lons'].append(max_hghtlons[j])
                            storms[sort_id[0]]['hght_min'].append(max_hght[j])
                            storms[sort_id[0]]['time'].append(valid_obj)
                            used_id.append(sort_id[0])
                            found = 1

                        #Otherwise, use distance & bearing matching criteria
                        else:
                            for k in sort_id:

                                if found == 1: continue
                                
                                if len(storms[k]['hght_lats']) < 2: break

                                #-------------------------------------------------------------------------------------
                                # 1. Calculate height track motion trajectory
                                #-------------------------------------------------------------------------------------

                                #Get last 2 time steps of coordinates
                                ilat = storms[k]['hght_lats'][-1]
                                ilon = storms[k]['hght_lons'][-1]
                                ilat_p = storms[k]['hght_lats'][-2]
                                ilon_p = storms[k]['hght_lons'][-2]

                                #Get 3 time step average if possible
                                if len(storms[k]['hght_lats']) > 2:
                                    ilat_p2 = storms[k]['hght_lats'][-3]
                                    ilon_p2 = storms[k]['hght_lons'][-3]

                                #-------------------------------------------------------------------------------------
                                # 2. Calculate vorticity track motion trajectory
                                #-------------------------------------------------------------------------------------

                                #Get last 2 time steps of coordinates
                                vlat = storms[k]['vort_lats'][-1]
                                vlon = storms[k]['vort_lons'][-1]
                                vlat_p = storms[k]['vort_lats'][-2]
                                vlon_p = storms[k]['vort_lons'][-2]

                                #Extrapolate motion vector forward
                                dist_last = great_circle((vlat_p,vlon_p), (vlat,vlon)).kilometers
                                bearing_last = calculate_initial_compass_bearing((vlat_p,vlon_p), (vlat,vlon))# + 180.0

                                destination = VincentyDistance(kilometers=dist_last).destination((vlat,vlon), bearing_last)
                                proj_lat, proj_lon = destination.latitude, destination.longitude

                                #Get 3 time step average if possible
                                if len(storms[k]['vort_lats']) > 2:
                                    vlat_p2 = storms[k]['vort_lats'][-3]
                                    vlon_p2 = storms[k]['vort_lons'][-3]

                                    dist_last2 = great_circle((vlat_p2,vlon_p2), (vlat_p,vlon_p)).kilometers
                                    bearing_last2 = calculate_initial_compass_bearing((vlat_p2,vlon_p2), (vlat_p,vlon_p))# + 180.0

                                    dist_last = (dist_last+dist_last2)/2.0
                                    bearing_last = avg_angle(bearing_last,bearing_last2)

                                    destination = VincentyDistance(kilometers=dist_last).destination((vlat,vlon), bearing_last)
                                    proj_lat, proj_lon = destination.latitude, destination.longitude

                                #///////////////////// MAP CODE /////////////////////
                                if save_map == 1:
                                    x1,y1 = m(vlon,vlat)
                                    x2,y2 = m(proj_lon,proj_lat)
                                    #print(f"{ilon},{ilat} ---- {proj_lon},{proj_lat}")
                                    plt.plot([x1,x2],[y1,y2],'-',linewidth=2.0,color='#ff00ff')
                                    plt.plot(x2,y2,'o',ms=10,color='#ff00ff')
                                #///////////////////// MAP CODE /////////////////////

                                #-------------------------------------------------------------------------------------
                                # 3. check if vort max is within 400km radius of projected location
                                #-------------------------------------------------------------------------------------

                                #Check if vort max is within 400km radius of projected location
                                dist_check = great_circle((max_lats[j],max_lons[j]), (proj_lat,proj_lon)).kilometers
                                if dist_check <= 400.0:

                                    storms[k]['vort_lats'].append(max_lats[j])
                                    storms[k]['vort_lons'].append(max_lons[j])
                                    storms[k]['vort_max'].append(max_vort[j])
                                    storms[k]['hght_lats'].append(max_hghtlats[j])
                                    storms[k]['hght_lons'].append(max_hghtlons[j])
                                    storms[k]['hght_min'].append(max_hght[j])
                                    storms[k]['time'].append(valid_obj)
                                    used_id.append(k)
                                    found = 1

                                #///////////////////// MAP CODE /////////////////////
                                if save_map == 1:
                                    bound_lats1 = []
                                    bound_lons1 = []
                                    for kk in np.arange(0,361):
                                        origin = geopy.Point(proj_lat,proj_lon) #second
                                        destination = great_circle(kilometers=400).destination(origin, kk)
                                        flat, flon = destination.latitude, destination.longitude
                                        bound_lats1.append(flat)
                                        bound_lons1.append(flon)
                                    x3,y3 = m(bound_lons1,bound_lats1)
                                    plt.plot(x3,y3,'-',color='yellow')
                                #///////////////////// MAP CODE /////////////////////

                    #-------------------------------------------------------------------------------------
                    #Extra check if cyclone still not found
                    #-------------------------------------------------------------------------------------

                    #elif len(max_vort) > 1:
                    if found == 0 and sort_disthght[0] <= 800.0:

                        #///////////////////// MAP CODE /////////////////////
                        if save_map == 1:
                            bound_lats1 = []
                            bound_lons1 = []
                            for kk in np.arange(0,361):
                                origin = geopy.Point(max_lats[j],max_lons[j]) #second
                                destination = great_circle(kilometers=1000).destination(origin, kk)
                                flat, flon = destination.latitude, destination.longitude
                                bound_lats1.append(flat)
                                bound_lons1.append(flon)
                            x3,y3 = m(bound_lons1,bound_lats1)
                            plt.plot(x3,y3,'-',color='#00ff00')

                            bound_lats1 = []
                            bound_lons1 = []
                            for kk in np.arange(0,361):
                                origin = geopy.Point(max_lats[j],max_lons[j]) #second
                                destination = great_circle(kilometers=750).destination(origin, kk)
                                flat, flon = destination.latitude, destination.longitude
                                bound_lats1.append(flat)
                                bound_lons1.append(flon)
                            x3,y3 = m(bound_lons1,bound_lats1)
                            plt.plot(x3,y3,'-',color='orange')
                        #///////////////////// MAP CODE /////////////////////

                        """
                        THIS IS THE STEP WE WANT TO REWRITE!!!!

                        Essentially, do this match, but add a motion & direction criteria for the height.
                        """

                        #-------------------------------------------------------------------------------------
                        # 1. check if this is the only cyclone within a 1000km radius
                        #-------------------------------------------------------------------------------------

                        sort_id = sort_id_full[sort_disthght <= 800.0]

                        max_id = []
                        max_disthghtcurr = []
                        max_distvortcurr = []

                        max_disthghtlast = []
                        max_distvortlast = []

                        #Identify distance from all CURRENT cyclones
                        for k in range(len(max_vort)):

                            if j == k: continue

                            #Check to see if new vort max is within similar location
                            max_disthghtcurr.append(great_circle((max_hghtlats[k],max_hghtlons[k]),(max_hghtlats[j],max_hghtlons[j])).kilometers)
                            max_distvortcurr.append(great_circle((max_lats[k],max_lons[k]),(max_lats[j],max_lons[j])).kilometers)

                            hlat = storms[sort_id[0]]['hght_lats'][-1]
                            hlon = storms[sort_id[0]]['hght_lons'][-1]
                            vlat = storms[sort_id[0]]['vort_lats'][-1]
                            vlon = storms[sort_id[0]]['vort_lons'][-1]
                            max_disthghtlast.append(great_circle((hlat,hlon),(max_hghtlats[j],max_hghtlons[j])).kilometers)
                            max_distvortlast.append(great_circle((vlat,vlon),(max_lats[j],max_lons[j])).kilometers)
                            max_id.append(k)

                        #Sort the distance from shortest to largest
                        max_disthghtcurr = np.array(max_disthghtcurr)
                        max_id = np.array(max_id)
                        max_distvortcurr = np.array(max_distvortcurr)
                        max_disthghtlast = np.array(max_disthghtlast)
                        max_distvortlast = np.array(max_distvortlast)

                        sort2_args = np.argsort(max_disthghtcurr)
                        sort2_disthght = (max_disthghtcurr[sort2_args])
                        sort2_distvort = (max_distvortcurr[sort2_args])
                        sort2_disthghtlast = (max_disthghtlast[sort2_args])
                        sort2_distvortlast = (max_distvortlast[sort2_args])
                        sort2_id = (max_id[sort2_args])

                        #Check if cyclone should be added
                        if len(max_vort) > 1 and sort2_distvort[0] > 1000 and sort2_disthght[0] > 1000 and sort_id[0] not in used_id and sort2_disthghtlast[0] < 750 and sort2_distvortlast[0] < 750:

                            #ADD CHECK FOR STORM MOTION!!! CHECK ALICIA'S CODE FOR THIS!!!!!

                            storms[sort_id[0]]['vort_lats'].append(max_lats[j])
                            storms[sort_id[0]]['vort_lons'].append(max_lons[j])
                            storms[sort_id[0]]['vort_max'].append(max_vort[j])
                            storms[sort_id[0]]['hght_lats'].append(max_hghtlats[j])
                            storms[sort_id[0]]['hght_lons'].append(max_hghtlons[j])
                            storms[sort_id[0]]['hght_min'].append(max_hght[j])
                            storms[sort_id[0]]['time'].append(valid_obj)
                            used_id.append(sort_id[0])
                            found = 1

                            #///////////////////// MAP CODE /////////////////////
                            if save_map == 1:
                                x4,y4 = m(storms[sort_id[0]]['hght_lats'][-1],storms[sort_id[0]]['hght_lons'][-1])
                                plt.plot(x4,y4,'o',ms=10,color='orange')
                            #///////////////////// MAP CODE /////////////////////

                        #-------------------------------------------------------------------------------------
                        # 2. If not, take previous cyclones, loop from smallest to largest, and do
                        #    steering motion matching criteria
                        #
                        # Here, use weighted sorting criteria
                        #-------------------------------------------------------------------------------------

                        else:

                            #Loop through all IDs
                            sort_id = weight_sort_id_full[(weight_sort_disthght <= 800.0) & (weight_sort_distvort <= 1500.0)]
                            sort_idvort = weight_sort_distvort[(weight_sort_disthght <= 800.0) & (weight_sort_distvort <= 1500.0)]

                            for sort_idx in range(len(sort_id)):
                                k = sort_id[sort_idx]

                                if found == 1: continue

                                #If 2 or more time steps available, use trajectory approach
                                if len(storms[k]['hght_lats']) >= 2:

                                    #Get last 2 time steps of coordinates
                                    ilat = storms[k]['hght_lats'][-1]
                                    ilon = storms[k]['hght_lons'][-1]
                                    ilat_p = storms[k]['hght_lats'][-2]
                                    ilon_p = storms[k]['hght_lons'][-2]

                                    #Extrapolate motion vector forward
                                    dist_last = great_circle((ilat_p,ilon_p), (ilat,ilon)).kilometers
                                    bearing_last = calculate_initial_compass_bearing((ilat_p,ilon_p), (ilat,ilon))# + 180.0

                                    destination = VincentyDistance(kilometers=dist_last).destination((ilat,ilon), bearing_last)
                                    proj_lat, proj_lon = destination.latitude, destination.longitude

                                    #Calculate bearing to this cyclone
                                    dist_now = great_circle((ilat,ilon), (max_hghtlats[j],max_hghtlons[j]))
                                    bearing_now = calculate_initial_compass_bearing((ilat,ilon), (max_hghtlats[j],max_hghtlons[j]))

                                    #Calculate difference in bearing between last & this vector
                                    diff_bearing = abs(angle_diff(bearing_last,bearing_now))

                                    #If bearing <= 60 degrees, match the cyclone
                                    if dist_now <= 200.0 or diff_bearing <= 60:

                                        storms[k]['vort_lats'].append(max_lats[j])
                                        storms[k]['vort_lons'].append(max_lons[j])
                                        storms[k]['vort_max'].append(max_vort[j])
                                        storms[k]['hght_lats'].append(max_hghtlats[j])
                                        storms[k]['hght_lons'].append(max_hghtlons[j])
                                        storms[k]['hght_min'].append(max_hght[j])
                                        storms[k]['time'].append(valid_obj)
                                        used_id.append(k)
                                        found = 1

                                #If 1 step only, then use distance criteria
                                else:

                                    if len(sort_idvort) > 1 and sort_idx < len(sort_idvort) - 1:
                                        if sort_idvort[sort_idx] < sort_idvort[sort_idx + 1]:
                                            storms[sort_id[0]]['vort_lats'].append(max_lats[j])
                                            storms[sort_id[0]]['vort_lons'].append(max_lons[j])
                                            storms[sort_id[0]]['vort_max'].append(max_vort[j])
                                            storms[sort_id[0]]['hght_lats'].append(max_hghtlats[j])
                                            storms[sort_id[0]]['hght_lons'].append(max_hghtlons[j])
                                            storms[sort_id[0]]['hght_min'].append(max_hght[j])
                                            storms[sort_id[0]]['time'].append(valid_obj)
                                            used_id.append(sort_id[0])
                                            found = 1




                #If >=750km, then there is no existing cyclone that is likely this, so add as a new cyclone
                if found == 0:
                    if max_vort[j] > vort_thres:
                        storms.append({"id":cid,"vort_lats":[max_lats[j]],"vort_lons":[max_lons[j]],"vort_max":[max_vort[j]],"hght_lats":[max_hghtlats[j]],"hght_lons":[max_hghtlons[j]],"hght_min":[max_hght[j]],"time":[valid_obj]})
                        cid += 1


            #///////////////////// MAP CODE /////////////////////
            if save_map == 1:
                #Plot tracks & current positions so far
                #fig,ax=plt.subplots(figsize=(15,12),dpi=125)
                tlon,tlat = np.meshgrid(lons,lats)
                x,y = m(tlon,tlat)
                #cs = plt.contourf(x,y,data.g.isel(lev=0)*0.1,np.arange(90,180,3.0))
                cs = plt.contourf(x,y,data.g.sel(lev=925)*0.1,np.arange(21,100,3))
                plt.title("Tracks through "+valid_str)
                m.colorbar(cs,size='1.8%', pad='0.8%')
                plt.contour(x,y,data_vort,[vort_thres],colors='#ff00ff',linewidths=1.1)
                for kk in range(len(storms)):
                    vort_lats = storms[kk]['vort_lats']
                    vort_lons = storms[kk]['vort_lons']
                    vort_max = storms[kk]['vort_max']
                    hght_lats = storms[kk]['hght_lats']
                    hght_lons = storms[kk]['hght_lons']
                    hght_min = storms[kk]['hght_min']
                    iter_time = storms[kk]['time']

                    x,y = m(vort_lons,vort_lats)
                    plt.plot(x,y,'-',linewidth=1.0,color='r')
                    if iter_time[-1] == valid_obj:
                        x,y = m(vort_lons[-1],vort_lats[-1])
                        plt.plot(x,y,'o',ms=10,color='r',mec='k',mew=0.5)

                    x,y = m(hght_lons,hght_lats)
                    plt.plot(x,y,'-',linewidth=1.0,color='b')
                    if iter_time[-1] == valid_obj:
                        x,y = m(hght_lons[-1],hght_lats[-1])
                        plt.plot(x,y,'o',ms=10,color='b',mec='k',mew=0.5)

                plt.plot([],color='#ff00ff',linewidth=1.1,label='min threshold of area-averaged vort')
                plt.plot([],'-o',color='r',linewidth=1.0,label='925-hPa vort track')
                plt.plot([],'-o',color='b',linewidth=1.0,label='925-hPa height track')
                plt.legend(loc=2)

                if valid_obj in storm_dates:
                    nidx = storm_dates.index(valid_obj)

                    nlat = storm_lat[0:nidx+1]
                    nlon = storm_lon[0:nidx+1]
                    nx,ny = m(nlon,nlat)
                    plt.plot(nx,ny,'-',linewidth=1.0,color='white')


                    nlat = storm_lat[nidx]
                    nlon = storm_lon[nidx]
                    nx,ny = m(nlon,nlat)
                    plt.plot(nx,ny,'o',ms=10,color='white',mec='k',mew=0.5)

                parallels = np.arange(-90,90,10)
                m.drawparallels(parallels,labels=[1,0,0,0],fontsize=9,linewidth=0.7)
                meridians = np.arange(0,360,10)
                m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=9,linewidth=0.7)

                save_str = dt.datetime.strftime(valid_obj,'%Y%m%d%H')
                plt.savefig(f"{save_str}.png")
                plt.close()
            #///////////////////// MAP CODE /////////////////////


    #=============================================================================================

    found_idx = -1
    min_dist = 9999

    #print storm_lat
    #print storm_lon
    
    #Numbers based on Alicia's research
    dist_thres = 0
    if leadtime == 0: dist_thres = 400
    if leadtime == 1: dist_thres = 600
    if leadtime == 2: dist_thres = 800
    if leadtime == 3: dist_thres = 1000
    if leadtime == 4: dist_thres = 1200
    if leadtime == 5: dist_thres = 1400

    for iz in np.arange(len(storms)):
        cur_id = storms[iz]['id']
        cur_lats = storms[iz]['hght_lats']
        cur_lons = storms[iz]['hght_lons']
        cur_vals = storms[iz]['hght_min']
        cur_time = storms[iz]['time']
        
        new_stormdate = [pd.to_datetime(ij) for ij in storm_date]

        if peak_time in cur_time and peak_time in new_stormdate:

            pk_idx = cur_time.index(peak_time)
            pk_lat = cur_lats[pk_idx]
            pk_lon = cur_lons[pk_idx]

            full_idx = new_stormdate.index(peak_time)
            point1 = (pk_lat,pk_lon)
            point2 = (storm_lat[full_idx],storm_lon[full_idx])
            dist = great_circle(point1,point2).kilometers

            if dist < dist_thres and len(cur_lats) >= 4: #used to be 700
                if dist < min_dist:
                    min_dist = dist
                    found_idx = iz
                    
        elif (peak_time+dt.timedelta(hours=6)) in cur_time and (peak_time+dt.timedelta(hours=6)) in new_stormdate:

            new_time = peak_time+dt.timedelta(hours=6)
            pk_idx = cur_time.index(new_time)
            pk_lat = cur_lats[pk_idx]
            pk_lon = cur_lons[pk_idx]

            full_idx = new_stormdate.index(new_time)
            point1 = (pk_lat,pk_lon)
            point2 = (storm_lat[full_idx],storm_lon[full_idx])
            dist = great_circle(point1,point2).kilometers

            if dist < dist_thres and len(cur_lats) >= 4: #used to be 700
                if dist < min_dist:
                    min_dist = dist
                    found_idx = iz
                    
        elif (peak_time-dt.timedelta(hours=6)) in cur_time and (peak_time-dt.timedelta(hours=6)) in new_stormdate:

            new_time = peak_time-dt.timedelta(hours=6)
            pk_idx = cur_time.index(new_time)
            pk_lat = cur_lats[pk_idx]
            pk_lon = cur_lons[pk_idx]

            full_idx = new_stormdate.index(new_time)
            point1 = (pk_lat,pk_lon)
            point2 = (storm_lat[full_idx],storm_lon[full_idx])
            dist = great_circle(point1,point2).kilometers

            if dist < dist_thres and len(cur_lats) >= 4: #used to be 700
                if dist < min_dist:
                    min_dist = dist
                    found_idx = iz
                    
        elif (peak_time+dt.timedelta(hours=12)) in cur_time and (peak_time+dt.timedelta(hours=12)) in new_stormdate and forward == 1:

            new_time = peak_time+dt.timedelta(hours=12)
            pk_idx = cur_time.index(new_time)
            pk_lat = cur_lats[pk_idx]
            pk_lon = cur_lons[pk_idx]

            full_idx = new_stormdate.index(new_time)
            point1 = (pk_lat,pk_lon)
            point2 = (storm_lat[full_idx],storm_lon[full_idx])
            dist = great_circle(point1,point2).kilometers

            if dist < dist_thres and len(cur_lats) >= 4: #used to be 700
                if dist < min_dist:
                    min_dist = dist
                    found_idx = iz
                    
        elif (peak_time-dt.timedelta(hours=12)) in cur_time and (peak_time-dt.timedelta(hours=12)) in new_stormdate and forward == 0:

            new_time = peak_time-dt.timedelta(hours=12)
            pk_idx = cur_time.index(new_time)
            pk_lat = cur_lats[pk_idx]
            pk_lon = cur_lons[pk_idx]

            full_idx = new_stormdate.index(new_time)
            point1 = (pk_lat,pk_lon)
            point2 = (storm_lat[full_idx],storm_lon[full_idx])
            dist = great_circle(point1,point2).kilometers

            if dist < dist_thres and len(cur_lats) >= 4: #used to be 700
                if dist < min_dist:
                    min_dist = dist
                    found_idx = iz

    #=============================================================================================

    if found_idx >= 0:

        mslp_lat = storms[found_idx]['hght_lats']
        mslp_lon = storms[found_idx]['hght_lons']
        mslp_val = storms[found_idx]['hght_min']
        mslp_vor = storms[found_idx]['vort_max']
        mslp_time = storms[found_idx]['time']

        #Convert to print-able output
        #ens;lon;lat;mslp;valid date
        line_lat = ((str(mslp_lat).replace("[","")).replace("]","")).replace(" ","")
        line_lon = ((str(mslp_lon).replace("[","")).replace("]","")).replace(" ","")
        line_val = ((str(mslp_val).replace("[","")).replace("]","")).replace(" ","")
        line_vor = ((str(mslp_vor).replace("[","")).replace("]","")).replace(" ","")
        #mslp_time = [dt.datetime.utcfromtimestamp(i.astype(int) * 1e-9) for i in mslp_time]
        mslp_time = [dt.datetime.strftime(i,'%Y%m%d%H') for i in mslp_time]
        line_time = (((str(mslp_time).replace("[","")).replace("]","")).replace(" ","")).replace("'","")
        line = str(ens) + ";" + line_lat + ";" + line_lon + ";" + line_val + ";" + line_vor + ";" + line_time
        saveline += line + "\n"

rundate = dt.datetime.strftime(init_date,'%Y%m%d%H')
print(saveline)
#o = open(output_path+input_id+"_"+rundate+".csv","w")
o = open(str(input_id)+"_"+rundate+".csv","w")
o.write(saveline)
o.close()

#========================================================================

#Close datasets
data.close()
print("Done with "+str(input_id)+" for leadtime "+str(leadtime))

#========================================================================

#IMPORT DATA FROM .DAT FILE
colnames1 = ['ens','latitude','longitude','height','vorticity','Date']

df2 = pd.read_csv(str(input_id)+"_"+rundate+".csv", sep=';',  header=None, names=colnames1)

#print(df2)

# CREATE XARRAT DATASET FROM PANDAS DATAFRAME
xar = xr.Dataset.from_dataframe(df2)

## add variable attribute metadata
xar['ens'].attrs={'units':'none', 'long_name':'Ensamble'}
xar['latitude'].attrs={'units':'degrees', 'long_name':'Latitude'}
xar['longitude'].attrs={'units':'degrees', 'long_name':'Longitude'}
xar['height'].attrs={'units':'millibar', 'long_name':'Height'}
xar['vorticity'].attrs={'units':'per second', 'long_name':'Vorticity Values'}
xar['Date'].attrs={'units':'time', 'long_name':'Date and Time'}

# add global attribute metadata
xar.attrs={'Conventions':'CF-1.6', 'title':'Tracking Data', 'summary':'Tracking Data generated'}

#print xr
print(xar)

# save to netCDF
xar.to_netcdf('test.nc')

print("Saved")

