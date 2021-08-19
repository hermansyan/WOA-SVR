# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 15:29:56 2021

@author: ASUS
"""

import csv
import numpy as np
import json
import time
import SVR_Model
#import mysql.connector as ms
#import mysql.connector as msql

#db = msql.connect(host="localhost",user="root",passwd="",database="estimasi")

# method read_csv for read csv file and save it to array

def read_csv(file_name):
    array_2D = []
    with open(file_name, 'r') as csvfile:
        read = csv.reader(csvfile, delimiter=';')
        for row in read:
#            array_2D.append(map(int,row))
            array_2D.append([int(i) for i in row])
    return array_2D

def split_data(data, proportion):
    dataTraining = data[0:int(np.floor(proportion*len(data)))]    
    dataTesting = data[int(np.floor(proportion*len(data))):len(data)]
    return dataTraining, dataTesting

# method get_max to find the maximum value of data
def get_max(data):
    max_value = -999
    for i in data:        
        for j in i:            
            if (j > max_value):
                max_value = j
    return max_value

# method get_min to find the minimum value of data
def get_min(data):
    min_value = 9999999999
    for i in data:
        for j in i:
            if (j < min_value):
                min_value = j
    return min_value

# method normalization to convert data to normalized value
def normalization(data, proportion):    
    res = np.zeros((len(data),len(data[0])),dtype=float)
    max_value = float(get_max(data))
    min_value = float(get_min(data))
    for i in range(len(res)):
        for j in range(len(res[i])):
            res[i][j] = (data[i][j] - min_value) / (max_value - min_value)    
    dataTraining = res[0:int(np.floor(proportion*len(data)))]    
    dataTesting = res[int(np.floor(proportion*len(data))):len(data)]
    return dataTraining, dataTesting

def decimal_normalization(data, proportion, komoditas):
    if(komoditas == "beras"):
        var_normalisasi = [6,7,5,6]
    elif(komoditas == "jagung"):
        var_normalisasi = [6,7,5,6]
    elif(komoditas == "kedelai"):
        var_normalisasi = [4,7,5,4]
    elif(komoditas == "bawang_merah"):
        var_normalisasi = [4,7,6,6]
    elif(komoditas == "cabe_besar"):
        var_normalisasi = [4,7,6,6]
    else:
        var_normalisasi = [4,7,6,6]
    #var_normalisasi = [4,7,5,4] // kedelai
    #var_normalisasi = [4,7,6,6]
    res = np.zeros((len(data),len(data[0])),dtype=float)
    #max_value = float(get_max(data))
    #min_value = float(get_min(data))
    for i in range(len(res)):
        for j in range(len(res[i])):
            #res[i][j] = (data[i][j] - min_value) / (max_value - min_value)    
            res[i][j] = float(data[i][j]) / pow(10,var_normalisasi[j])
            #print(res[i][j])
    dataTraining = res[0:int(np.floor(proportion*len(data)))]    
    dataTesting = res[int(np.floor(proportion*len(data))):len(data)]
    return dataTraining, dataTesting

# method dist to search the distance between all elements
def get_dist(data):
    distance = np.zeros((len(data),len(data)),dtype=float)
    for i in range(len(distance)):
        for j in range(len(distance)):
            distance[i][j] = calc_dist(data[i],data[j])
    return distance

# method calc_dist to calculate the distance between two array
def calc_dist(array1, array2):
    res = 0.0
    for i in range(len(array1) - 1):
        res += np.power((array1[i] - array2[i]),2)
    return res

# method get_kernel_rbf to calculate the value of kernel RBF from data training
def get_kernel_rbf(data_dist,sigma):
    kernel = np.zeros_like(data_dist)
    for i in range(len(data_dist)):
        for j in range(len(data_dist[i])):
            kernel[i][j] = np.exp(-(data_dist[i][j]/(2*np.power(sigma,2))))
    return kernel

def get_hessian(kernel_data,lamda):
    hessian = np.zeros_like(kernel_data)
    for i in range(len(kernel_data)):
        for j in range(len(kernel_data[i])):
            hessian[i][j] = kernel_data[i][j] + np.power(lamda,2)
    return hessian

def calc_MSE(prediction, actual):
    res = np.zeros_like(prediction)
    for i in range(len(prediction)):
        res[i] = np.power((actual[i] - prediction[i]),2)
    return np.average(res)

def calc_MAE(prediction, actual):
    res = np.zeros_like(prediction)
    for i in range(len(prediction)):
        res[i] = np.abs(actual[i] - prediction[i])
    return np.average(res)

# def update_db(komoditas, d_alpha):
#     conn = ms.connect(user='root', password='', host='localhost', database='estimasi')
#     cursor = conn.cursor()
#     str_alpha = "alpha_"+komoditas
#     queryDelete = """Delete from """+str_alpha
#     cursor.execute(queryDelete)
#     add_alpha = """INSERT INTO """+str_alpha+""" VALUES (%s)"""
#     #print(add_alpha)
    
#     #alpha = [2, -2.000338880028326, -1.7484869882343739, -3.743523156579997, 1.7120601398098891, 7.1781298102175075, 4.256270656989639, 0.9214269066068619, -9.542214992186763, 0.2558967213865766, -1.6131547996969597, 0.10469222399301072, 3.514537248721711, -0.15861476503715854]
#     for alpha_i in d_alpha:
#         #print(alpha_i)
#         cursor.execute(add_alpha,((alpha_i.tolist()),))
#     conn.commit()
#     conn.close()

# def select_db(komoditas):    
#     conn = ms.connect(user='root', password='', host='localhost', database='estimasi')
#     cursor = conn.cursor()
#     querySelect = """SELECT * FROM """+ komoditas    
#     cursor.execute(querySelect)
#     dataKomoditas = []
#     res_tahun = []
#     for (tahun, luas_tanam, jml_penduduk, luas_lahan, produksi) in cursor:
#         dataKomoditas.append([luas_tanam, jml_penduduk, luas_lahan, produksi])
#         res_tahun.append(tahun)
#         #print("{}, {}, {}, {}, {}".format(tahun,luas_tanam,jml_penduduk,luas_lahan, produksi))
#     return dataKomoditas, res_tahun
#     conn.close()


# MAIN
start_time = time.time()
C_value = 100
cLR = 0.05
#epsilon = 0.00001
epsilon = 0.0005
sigma = 0.3
lamda = 0.1
iter_max =100
dataTraining = []
dataTesting = []
alpha = []
alpha_star = []
max_data = 0
min_data = 0
y_prediksi = []
prop = 0.8

init_model = SVR_Model.SVR_Model()
init_model.cLR = cLR
init_model.C_value = C_value
init_model.epsilon = epsilon
init_model.lamda = lamda
init_model.sigma = sigma

def exec_svr_woa(model=init_model, is_print_log=False):
    # return 0;

    dataAll = read_csv("data/dataCovid.csv")
    
    dataTraining, dataTesting = normalization(dataAll, 0.7)
    x_training = ((np.array(dataTraining))[:,:-1]).tolist()
    x_testing = ((np.array(dataTesting))[:,:-1]).tolist()
    y_training = ((np.array(dataTraining))[:,-1]).tolist()
    y_testing = ((np.array(dataTesting))[:,-1]).tolist()
    
    jarak = get_dist(dataTraining)
    #for data in jarak:
        #print(data)
    
    kernel = get_kernel_rbf(jarak, model.sigma)
    #for data in kernel:
        #print(data)
    
    hessian_matrix = get_hessian(kernel, model.lamda)
    #for data in hessian_matrix:
        #print(data)
    
    # SEQUENTIAL LEARNING
    
    # Step 1 : Initialize alpha and alpha_star with 0
    alpha = [0] * len(dataTraining)
    alpha_star = [0] * len(dataTraining)
    E_value = [0] * len(dataTraining)
    delta_alpha = [0.0] * len(dataTraining)
    delta_alpha_star = [0.0] * len(dataTraining)
    gamma = model.cLR / get_max(hessian_matrix)
    
    y_prediksi = [0.0] * len(dataTraining)
    
    # Step 2 : For each training point, compute :
    x = 0
    min_mse = 999999
    iterate = True
    
    while(iterate):
        print(x) if is_print_log else ''
        # 2.1 : Compute Ei
        y = np.transpose(dataTraining)[3]
        for i in range(len(jarak)):
            sum_prod = np.sum([(alp_s - alp)*H for H,alp_s,alp in zip(hessian_matrix[i], alpha_star, alpha)])
            E_value[i] = y[i] - sum_prod
            
        
        # 2.2 : Compute delta alpha and delta alpha star    
        delta_alpha_star = [min(max(gamma*(E - model.epsilon), -A), model.C_value - A) for E,A in zip(E_value, alpha_star)]
        delta_alpha = [min(max(gamma*(-E - model.epsilon), -A), model.C_value - A) for E,A in zip(E_value, alpha)]
        
        
        # 2.3 : Compute new alpha and alpha star
        alpha = [a + b for a,b in zip(alpha, delta_alpha)]
        alpha_star = [a + b for a,b in zip(alpha_star, delta_alpha_star)]
        
        for i in range(len(y_prediksi)):
            y_prediksi[i] = np.sum( [(alp_s - alp)*H for H,alp_s,alp in zip(hessian_matrix[i],alpha_star,alpha)])
            
        if(((max(delta_alpha_star) < model.epsilon) and (max(delta_alpha) < model.epsilon)) or (x > iter_max)):
            d_alpha = [a-b for a,b in zip(alpha_star, alpha)]
            iterate = False
            
        if (min_mse > calc_MSE(y_prediksi, y)):
            min_mse = calc_MSE(y_prediksi, y)
        else:
            iterate = False
            print("Iterasi ke-" + str(x-1)) if is_print_log else ''
            
        x = x+1
    print("ITERASI = "+str(x)) if is_print_log else ''
    
    max_data = float(get_max(dataAll))
    min_data = float(get_min(dataAll))
    y_denorm = np.zeros_like(y_prediksi)
    for i in range(len(y_prediksi)):
        y_denorm[i] = y_prediksi[i] * (max_data-min_data) + min_data
        
    print(delta_alpha_star) if is_print_log else ''
    print(delta_alpha) if is_print_log else ''
    print(min_mse) if is_print_log else ''
    
    model.last_iteration = x
    model.errors = min_mse
    model.alpha_function = [(alp_s - alp) for alp_s,alp in zip(alpha_star,alpha)]
    
    model.training_data, _ = split_data(dataAll, 0.7)
    model.training_data = (np.array(model.training_data))[:,-1]
    model.validation_result = y_denorm
    
    return model

print(exec_svr_woa())
result = exec_svr_woa()
    
