# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 11:29:44 2017

@author: Hardik Galiawala
"""
import math 
import pandas as pd
import datetime
import gpxpy.geo as gpx


def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    #lat1 = math.radians(pointA[0])
    #lat2 = math.radians(pointB[0])
    
    lat1 = pointA[0]
    lat2 = pointB[0]
    #diffLong = math.radians(pointB[1] - pointA[1])
    diffLong = pointB[1] - pointA[1]
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

def set_haversine_distance(data_frame):
    data_frame['haversine_dist'] = [gpx.haversine_distance(data_frame.latitude[i], data_frame.longitude[i], data_frame.latitude[i+1], data_frame.longitude[i+1]) if i != (len(data_frame.t_user_id) - 1) else 0 for i in range(len(data_frame.t_user_id))]
    data_frame['haversine_dist'] = [(data_frame.haversine_dist[i] * data_frame.user_separator[i] * data_frame.day_separator[i]) for i in range(len(data_frame.t_user_id))]

def set_time_difference(data_frame):
    data_frame['time_difference'] = [(data_frame.collected_time[i+1] - data_frame.collected_time[i]).total_seconds() if i != (len(data_frame.t_user_id) - 1) else 0 for i in range(len(data_frame.t_user_id))]
    #data_frame['time_difference'] = [((datetime.combine(date.min, data_frame.time[i+1]) - datetime.combine(date.min, data_frame.time[i])).total_seconds()) if i != (len(data_frame.t_user_id) - 1) else 0 for i in range(len(data_frame.t_user_id))]
    data_frame['time_difference'] = [(data_frame.time_difference[i] * data_frame.user_separator[i] * data_frame.day_separator[i]) for i in range(len(data_frame.t_user_id))]

def set_speed(data_frame):
    data_frame['speed'] = [data_frame.haversine_dist[i] / (data_frame.time_difference[i]+ 0.1**10) for i in range(len(data_frame.t_user_id))]
    data_frame['speed'] = [(data_frame.speed[i] * data_frame.user_separator[i] * data_frame.day_separator[i]) for i in range(len(data_frame.t_user_id))]

def set_acceleration(data_frame):
    data_frame['acc'] = [(data_frame.haversine_dist[i+1] - data_frame.haversine_dist[i]) / (data_frame.time_difference[i]+ 0.1**10) if i != (len(data_frame.t_user_id) - 1) else 0 for i in range(len(data_frame.t_user_id))]
    data_frame['acc'] = [(data_frame.acc[i] * data_frame.user_separator[i] * data_frame.day_separator[i]) for i in range(len(data_frame.t_user_id))]

def set_bearing(data_frame):
    data_frame['bearing'] = [(calculate_initial_compass_bearing(data_frame.tup[i], data_frame.tup[i+1]) * data_frame.user_separator[i] * data_frame.day_separator[i]) if i != (len(data_frame.t_user_id) - 1) else 0 for i in range(len(data_frame.t_user_id))]
    
def main():
    print( " Main start")    
    print(datetime.datetime.now())
    #Reading data from csv file
    raw_data = pd.read_csv('C:/geolife_raw.csv')    
    #Converting date from string to date times
    raw_data['collected_time'] = pd.to_datetime(raw_data['collected_time'])
    #raw_data['latitude'] = np.radians(raw_data['latitude']) #[i] for i in range(len(raw_data.t_user_id)))
    #raw_data['longitude'] = np.radians(raw_data['longitude'])
    raw_data['date'] = [d.date() for d in raw_data['collected_time']]
    #User separator
    raw_data['user_separator'] = [0 if i == (len(raw_data.t_user_id) - 1) or raw_data.t_user_id[i] != raw_data.t_user_id[i+1] else 1 for i in range(len(raw_data.t_user_id))]
    print("User-separator completed")
    print(datetime.datetime.now())
    raw_data['day_separator'] = [0 if i == (len(raw_data.t_user_id) - 1) or raw_data.date[i] != raw_data.date[i+1] else 1 for i in range(len(raw_data.t_user_id))]
    print("Day separator completed")    
    print(datetime.datetime.now())    
    #Creating tuple of Latitude and Longitude as it is needed for calculating haversine distance
    raw_data['tup'] = [(raw_data.latitude[i], raw_data.longitude[i]) for i in range(len(raw_data.t_user_id))]
        
    print(datetime.datetime.now())
    set_haversine_distance(raw_data)
    print(datetime.datetime.now())
    set_time_difference(raw_data)
    print("TimeDiff completed ")
    print(datetime.datetime.now())
    set_speed(raw_data)
    print(datetime.datetime.now())
    set_acceleration(raw_data)
    print(datetime.datetime.now())
    set_bearing(raw_data)
    print(" Bearing completed")
    print(datetime.datetime.now())    
    print('Done')
    #Writing data to a file
    raw_data.to_csv('D:/geolifeA2.csv', header = True, index = False, mode = 'a')

main()