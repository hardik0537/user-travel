# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 10:22:49 2017

@author: Hardik Galiawala
"""


import pandas as pd
import datetime
import seaborn as sns
from numpy import median


def main():
    print( " Main start")    
    print(datetime.datetime.now())
    # Reading data from csv file
    raw_data = pd.read_csv('D:/geolifeA3.csv')
    #Preparing data for plotting
    df_hd = raw_data[['transportation_mode', 'haversine_dist_mean', 'haversine_dist_median', 'haversine_dist_max', 'haversine_dist_min']]
    df_speed = raw_data[['transportation_mode', 'speed_mean', 'speed_median', 'speed_max', 'speed_min', 'speed_std']]
    df_acc = raw_data[['transportation_mode', 'acc_mean', 'acc_median', 'acc_max', 'acc_min', 'acc_std']]
    df_bearing = raw_data[['transportation_mode', 'bearing_mean', 'bearing_median', 'bearing_max', 'bearing_min', 'bearing_std']]
    
    #Plotting graphs
    print("Transportation mode on X-axis v/s Median of Speed-Median on Y-axis")
    speed_median = sns.barplot(df_speed.transportation_mode, df_speed.speed_median, estimator = median)
    sns.despine(ax = speed_median) 
    print("Transportation mode on X-axis v/s Median of Distance-Median on Y-axis")
    haversine_median = sns.barplot(df_hd.transportation_mode, df_hd.haversine_dist_median, estimator = median)
    sns.despine(ax = haversine_median)
    
    
main()