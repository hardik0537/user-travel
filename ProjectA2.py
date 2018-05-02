# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:20:57 2017

@author: Hardik Galiawala
"""

import math 
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import gpxpy.geo as gpx


#Reference of this function
# https://stackoverflow.com/questions/20613396/selecting-and-deleting-columns-with-the-same-name-in-pandas-dataframe
def remove_dup_columns(frame):
     keep_names = set()
     keep_icols = list()
     for icol, name in enumerate(frame.columns):
          if name not in keep_names:
               keep_names.add(name)
               keep_icols.append(icol)
     return frame.iloc[:, keep_icols]
 

def main():
    print( " Main start")    
    print(datetime.datetime.now())
    # Reading data from csv file
    raw_data2 = pd.read_csv('D:/geolifeA2.csv')
    raw_data2 = raw_data2[['t_user_id', 'date', 'transportation_mode', 'user_separator', 'day_separator',  'haversine_dist', 'speed', 'acc', 'bearing']]
    raw_data2.date = pd.to_datetime(raw_data2.date)
    print("Discarding rows")
    print(datetime.datetime.now())
    #Discarding trajectories which are not useful
    raw_data2 = raw_data2[raw_data2.user_separator == 1]
    raw_data2 = raw_data2[raw_data2.day_separator == 1]
    # Removing classes as specified in the project requirement
    raw_data2 = raw_data2[raw_data2.transportation_mode != "run"]
    raw_data2 = raw_data2[raw_data2.transportation_mode != "motorcycle"]
    print("Completed")
    print(datetime.datetime.now())
    # Grouping data per user, day and transportation mode
    print("Group by started")
    raw_data_mean = raw_data2.groupby(['t_user_id', 'date', 'transportation_mode']).mean().reset_index()
    raw_data_mean.rename(columns = {'speed':'speed_mean', 'acc':'acc_mean', 'haversine_dist':'haversine_dist_mean', 'bearing':'bearing_mean'}, inplace = True)
    raw_data_min = raw_data2.groupby(['t_user_id', 'date', 'transportation_mode']).min().reset_index()
    raw_data_min.rename(columns = {'speed':'speed_min', 'acc':'acc_min', 'haversine_dist':'haversine_dist_min', 'bearing':'bearing_min'}, inplace = True)
    raw_data_max = raw_data2.groupby(['t_user_id', 'date', 'transportation_mode']).max().reset_index()
    raw_data_max.rename(columns = {'speed':'speed_max', 'acc':'acc_max', 'haversine_dist':'haversine_dist_max', 'bearing':'bearing_max'}, inplace = True)
    raw_data_median = raw_data2.groupby(['t_user_id', 'date', 'transportation_mode']).median().reset_index()
    raw_data_median.rename(columns = {'speed':'speed_median', 'acc':'acc_median', 'haversine_dist':'haversine_dist_median', 'bearing':'bearing_median'}, inplace = True)
    raw_data_std = raw_data2.groupby(['t_user_id', 'date', 'transportation_mode']).std().reset_index()
    raw_data_std.rename(columns = {'speed':'speed_std', 'acc':'acc_std', 'haversine_dist':'haversine_dist_std', 'bearing':'bearing_std'}, inplace = True)
    raw_data_count = raw_data2.groupby(['t_user_id', 'date', 'transportation_mode'])[['acc']].count().reset_index()
    raw_data_count.rename(columns = {'acc':'count_per_group'}, inplace = True)
    print("Completed")
    print(datetime.datetime.now())
    print("Concatenating")
    df_conc = pd.concat([raw_data_mean, raw_data_min, raw_data_max, raw_data_median, raw_data_std, raw_data_count], axis = 1, join = 'inner')
    print("Discarding trajectories < 10")
    print(datetime.datetime.now())
    df_conc = df_conc[df_conc.count_per_group > 10]
    print("Completed")
    print("Removing duplicate rows")
    df_agg = remove_dup_columns(df_conc)
    df_agg.to_csv('D:/geolifeA3.csv', header = True, index = False, mode = 'a')
main()