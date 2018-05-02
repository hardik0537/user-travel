# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 15:46:24 2017

@author: Hardik Galiawala
"""

import pandas as pd
import numpy as np
import scipy as sc
import plotly.plotly as py
import plotly.graph_objs as go

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

#creating object for Decision Tree
decision_tree = DecisionTreeClassifier()
#creating object for Random Forest 
random_forest = RandomForestClassifier(n_estimators = 20)


        
def main():
# Reading data from csv file
    raw_data = pd.read_csv('D:/geolifeA3.csv')    
    df_recon = raw_data[['haversine_dist_mean','speed_mean', 'acc_mean', 'bearing_mean', 'haversine_dist_min', 'speed_min', 'acc_min', 'bearing_min', 'haversine_dist_max',	'speed_max', 'acc_max',	'bearing_max',	'haversine_dist_median', 'speed_median', 'acc_median', 'bearing_median', 'haversine_dist_std', 'speed_std', 'acc_std', 'bearing_std', 'transportation_mode']]
    df_recon = df_recon.values
    
    #
    # Converting to array
    features = np.array(df_recon[:,:20])
    
    t_mode_class = np.array(df_recon[:,-1:]).ravel()
    
    # 10 fold cross validation for random forest with 20 estimators
    cross_val_dt_t_mode = cross_val_score(decision_tree, features, t_mode_class, cv=10, scoring = 'accuracy')    
    print ('The accuracy of Decision Tree Flat structure is : %.2f' %(cross_val_dt_t_mode.mean()))
    
    # 10 fold cross validation for random forest with 20 estimators
    cross_val_rf_t_mode = cross_val_score(random_forest, features, t_mode_class, cv=10, scoring = 'accuracy')
    print ('The accuracy of Random Forest Flat structure is : %.2f' %(cross_val_rf_t_mode.mean()))
    
    # Statistical t-test for flat classification
    stats_flat, p_val_flat = sc.stats.ttest_rel(cross_val_dt_t_mode, cross_val_rf_t_mode, axis = 0)
    
    if(p_val_flat <= 0.05):
        print("Good P-value, hence data did not occur by chance.")
        print(p_val_flat)
    else:
        print("Bad P-value")
        print(p_val_flat)

    #Binarizing the classes by heirarchy classfication
    target_rail = [1 if t_mode_class[i].lower() == 'train' 
                   or t_mode_class[i].lower() == 'subway' else 0 for i in range(len(t_mode_class))]
    target_on_road = [1 if t_mode_class[i].lower() == 'bus' 
                     or t_mode_class[i].lower() == 'taxi' 
                     or t_mode_class[i].lower() == 'car' 
                     or t_mode_class[i].lower() == 'walk' else 0 for i in range(len(t_mode_class))]
    target_on_wheels = [1 if t_mode_class[i].lower() == 'bus'
                        or t_mode_class[i].lower() == 'taxi' 
                        or t_mode_class[i].lower() == 'car' else 0 for i in range(len(t_mode_class))]
    target_walk = [1 if t_mode_class[i].lower() == 'walk' else 0 for i in range(len(t_mode_class))]
    target_bus = [1 if t_mode_class[i].lower() == 'bus' else 0 for i in range(len(t_mode_class))]
    target_taxi = [1 if t_mode_class[i].lower() == 'taxi' else 0 for i in range(len(t_mode_class))]
    target_car = [1 if t_mode_class[i].lower() == 'car' else 0 for i in range(len(t_mode_class))]
    target_subway = [1 if t_mode_class[i].lower() == 'subway' else 0 for i in range(len(t_mode_class))]
    target_train = [1 if t_mode_class[i].lower() == 'train' else 0 for i in range(len(t_mode_class))]
    
    '''
    # Splitting data into training and testing (70:30)
    train_walk_x, test_walk_x, train_walk_y, test_walk_y = train_test_split(features, target_walk, train_size = 0.7)
    train_other_x, test_other_x, train_other_y, test_other_y = train_test_split(features, target_other, train_size = 0.7)
    train_rail_x, test_rail_x, train_rail_y, test_rail_y = train_test_split(features, target_rail, train_size = 0.7)
    train_w_rail_x, test_w_rail_x, train_w_rail_y, test_w_rail_y = train_test_split(features, target_w_rail, train_size = 0.7)
    train_bus_x, test_bus_x, train_bus_y, test_bus_y = train_test_split(features, target_bus, train_size=0.7)
    train_taxi_x, test_taxi_x, train_taxi_y, test_taxi_y = train_test_split(features, target_taxi, train_size=0.7)
    train_car_x, test_car_x, train_car_y, test_car_y = train_test_split(features, target_car, train_size=0.7)
    train_subway_x, test_subway_x, train_subway_y, test_subway_y = train_test_split(features, target_subway, train_size=0.7)
    train_train_x, test_train_x, train_train_y, test_train_y = train_test_split(features, target_train, train_size=0.7)
    '''
    
    # Decision Tree - 10 fold cross validation 
    cross_val_dt_rail = cross_val_score(decision_tree, features, target_rail, cv=10, scoring = 'accuracy')
    cross_val_dt_on_road = cross_val_score(decision_tree, features, target_on_road, cv=10, scoring = 'accuracy')
    cross_val_dt_on_wheels = cross_val_score(decision_tree, features, target_on_wheels, cv=10, scoring = 'accuracy')
    cross_val_dt_walk = cross_val_score(decision_tree, features, target_walk, cv=10, scoring = 'accuracy')
    cross_val_dt_bus = cross_val_score(decision_tree, features, target_bus, cv=10, scoring = 'accuracy')
    cross_val_dt_taxi = cross_val_score(decision_tree, features, target_taxi, cv=10, scoring = 'accuracy')
    cross_val_dt_car = cross_val_score(decision_tree, features, target_car, cv=10, scoring = 'accuracy')
    cross_val_dt_subway = cross_val_score(decision_tree, features, target_subway, cv=10, scoring = 'accuracy')
    cross_val_dt_train = cross_val_score(decision_tree, features, target_train, cv=10, scoring = 'accuracy')
    
    
    # Decision Tree - Concatenating all accuracies from different hierarchies after cross validation stratification
    cross_val_dt_hierarchy = np.concatenate((cross_val_dt_walk, 
                                             cross_val_dt_on_road,
                                             cross_val_dt_rail, 
                                             cross_val_dt_on_wheels, 
                                             cross_val_dt_bus, 
                                             cross_val_dt_taxi, 
                                             cross_val_dt_car, 
                                             cross_val_dt_subway, 
                                             cross_val_dt_train), axis = 0)
    
    print ('The accuracy of Decision Tree Hierarchical structure is : %.2f' %(cross_val_dt_hierarchy.mean()))
    
    # Random Forest - 10 fold cross validation 
    cross_val_rf_rail = cross_val_score(random_forest, features, target_rail, cv=10, scoring = 'accuracy')
    cross_val_rf_on_road = cross_val_score(random_forest, features, target_on_road, cv=10, scoring = 'accuracy')
    cross_val_rf_on_wheels = cross_val_score(random_forest, features, target_on_wheels, cv=10, scoring = 'accuracy')
    cross_val_rf_walk = cross_val_score(random_forest, features, target_walk, cv=10, scoring = 'accuracy')
    cross_val_rf_bus = cross_val_score(random_forest, features, target_bus, cv=10, scoring = 'accuracy')
    cross_val_rf_taxi = cross_val_score(random_forest, features, target_taxi, cv=10, scoring = 'accuracy')
    cross_val_rf_car = cross_val_score(random_forest, features, target_car, cv=10, scoring = 'accuracy')
    cross_val_rf_subway = cross_val_score(random_forest, features, target_subway, cv=10, scoring = 'accuracy')
    cross_val_rf_train = cross_val_score(random_forest, features, target_train, cv=10, scoring = 'accuracy')
    
    #Random Forest Concatenating all accuracies from different hierarchies after cross validation stratification
    cross_val_rf_hierarchy = np.concatenate((cross_val_rf_walk, 
                                             cross_val_rf_on_road,
                                             cross_val_rf_rail, 
                                             cross_val_rf_on_wheels, 
                                             cross_val_rf_bus, 
                                             cross_val_rf_taxi, 
                                             cross_val_rf_car, 
                                             cross_val_rf_subway, 
                                             cross_val_rf_train), axis = 0)
    print ('The accuracy of Random Forest Hierarchical structure is : %.2f' %(cross_val_rf_hierarchy.mean()))
    
    # Statistical t-test for hierarchy classification
    stats_hierarchy, p_val_hierarchy = sc.stats.ttest_rel(cross_val_dt_hierarchy, cross_val_rf_hierarchy, axis = 0)
    
    if(p_val_flat <= 0.05):
        print("Good P-value, hence data did not occur by chance.")
        print(p_val_flat)
    else:
        print("Bad P-value")
        print(p_val_flat)
    

    y1 = [cross_val_rf_t_mode.mean(), cross_val_dt_t_mode.mean(), cross_val_rf_hierarchy.mean(), cross_val_dt_hierarchy.mean()]
    # Comparing heirarchical structure 
    data = [go.Bar( x=['Random Forest (Flat)', 'Decision Tree (Flat)', 'Random Forest (H)', 'Decision Tree (H)'], y = y1)]

    py.plot(data, filename='accuracy-Bar') 
    
    
main()