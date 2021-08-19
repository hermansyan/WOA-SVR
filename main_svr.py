# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 19:06:38 2021

@author: MUZANNI
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from random import random
import training_svr
import SVR_Model

NUMBER_OF_SOLUTION = 10
NUMBER_OF_ITERATIONS = 1
THRESHOLD_UPDATE_PROBABILITY = 0.5
b = 2

LOWER_BOUND_cLR = 0.01
UPPER_BOUND_cLR = 0.1
LOWER_BOUND_C = 10
UPPER_BOUND_C = 200
LOWER_BOUND_EPSILON = 0.0001
UPPER_BOUND_EPSILON = 0.1
LOWER_BOUND_LAMBDA = 0.05
UPPER_BOUND_LAMBDA = 0.5
LOWER_BOUND_SIGMA = 0.1
UPPER_BOUND_SIGMA = 0.5


fileName = 'data/dataCovid.csv'
rawFile = pd.read_csv(fileName, header=None, delimiter=";")

inputs = np.array(rawFile[rawFile.columns[:-1]])
outputs = np.array(rawFile[rawFile.columns[-1:]])

# Get Fitness of list of solution / agent
# def get_fitness(solutions):    
#     outputs_predict = np.dot(inputs, solutions)
#     errors = np.array([(i - j) ** 2 for i, j in zip(outputs, outputs_predict)]).sum() / len(outputs)
#     return {
#         'errors': errors,
#         'solutions': solutions
#         }

def get_fitness(solutions):
    prediction_model = SVR_Model.SVR_Model()
    prediction_model.cLR = solutions[0]
    prediction_model.C_value = solutions[1]
    prediction_model.epsilon = solutions[2]
    prediction_model.lamda = solutions[3]
    prediction_model.sigma = solutions[4]
    
    trained_model = training_svr.exec_svr_woa(model=prediction_model)
    
    return trained_model

# def get_random(multiplier = 1):
#     return random()*multiplier

def get_random(parameter = 'cLR'):
    switcher = {
        'cLR': np.random.uniform(LOWER_BOUND_cLR, UPPER_BOUND_cLR),
        'C_Value': np.random.uniform(LOWER_BOUND_C, UPPER_BOUND_C),
        'epsilon': np.random.uniform(LOWER_BOUND_EPSILON, UPPER_BOUND_EPSILON),
        'lamda': np.random.uniform(LOWER_BOUND_LAMBDA, UPPER_BOUND_LAMBDA),
        'sigma': np.random.uniform(LOWER_BOUND_SIGMA, UPPER_BOUND_SIGMA),
    }
    
    return switcher.get(parameter, np.random.uniform(LOWER_BOUND_cLR, UPPER_BOUND_cLR))

def get_best_solution(solutions):
    fitness_list = [get_fitness(solutions[:,idx]) for idx in range(len(solutions[0]))]
    sorted_fitness_list = sorted(fitness_list, key=lambda fitness: fitness.errors)
    return sorted_fitness_list[0]

def validate_range_solution(current_solution):
    validated_solution = []
    if (current_solution[0] < LOWER_BOUND_cLR):
        validated_solution.append(LOWER_BOUND_cLR)
    elif (current_solution[0] > UPPER_BOUND_cLR):
        validated_solution.append(UPPER_BOUND_cLR)
    else:
        validated_solution.append(current_solution[0])
    
    if (current_solution[1] < LOWER_BOUND_C):
        validated_solution.append(LOWER_BOUND_C)
    elif (current_solution[1] > UPPER_BOUND_C):
        validated_solution.append(UPPER_BOUND_C)
    else:
        validated_solution.append(current_solution[1])
    
    if (current_solution[2] < LOWER_BOUND_EPSILON):
        validated_solution.append(LOWER_BOUND_EPSILON)
    elif (current_solution[2] > UPPER_BOUND_EPSILON):
        validated_solution.append(UPPER_BOUND_EPSILON)
    else:
        validated_solution.append(current_solution[2])
    
    if (current_solution[3] < LOWER_BOUND_LAMBDA):
        validated_solution.append(LOWER_BOUND_LAMBDA)
    elif (current_solution[3] > UPPER_BOUND_EPSILON):
        validated_solution.append(UPPER_BOUND_EPSILON)
    else:
        validated_solution.append(current_solution[3])
    
    if (current_solution[4] < LOWER_BOUND_SIGMA):
        validated_solution.append(LOWER_BOUND_SIGMA)
    elif (current_solution[4] > UPPER_BOUND_SIGMA):
        validated_solution.append(UPPER_BOUND_SIGMA)
    else:
        validated_solution.append(current_solution[4])
        
    return validated_solution

model_svr = SVR_Model.SVR_Model()
model_svr.cLR = 0.05
model_svr.C_value = 100.0
model_svr.epsilon = 0.0005
model_svr.lamda = 0.1
model_svr.sigma = 0.3

# print(training_svr.exec_svr_woa(model=model_svr))

# Generate Random Solution
# row 0 -> cLR
# row 1 -> C_Value
# row 2 -> epsilon
# row 3 -> lamda
# row 4 -> sigma

solutions = np.array([
    [get_random('cLR'), 
     get_random('C_Value'),
     get_random('epsilon'),
     get_random('lamda'),
     get_random('sigma')] for i in range(NUMBER_OF_SOLUTION)]).transpose()

# # 
# fitness = get_fitness(solutions)

# Get best solution from all of solutions
best_solution = get_best_solution(solutions)

init_solution = solutions[:]
for idx_iter, current_iter in enumerate(range(NUMBER_OF_ITERATIONS)):
    print("ITERASI KE-", idx_iter)
    updated_solution_list = []
    for curr_solution in [solutions[:, idx] for idx in range(len(solutions[0]))]:
        # print(solution)
        update_probability = random()
        print(update_probability)
        updated_solution = []
        a = ((current_iter+1) / NUMBER_OF_ITERATIONS) * 2
        if (update_probability < THRESHOLD_UPDATE_PROBABILITY):            
            r = random()
            A = (2*a*r) - a
            C = 2*r
            if (abs(A) < 1):                
                print('ENCIRCLING')
                D = abs((C*best_solution.solutions) - curr_solution)
                updated_solution = best_solution.solutions - (A*D)
                
                
            else:
                print('SEARCHING')
                random_solution = solutions[:, int(random()*len(solutions))]
                D = abs((C*random_solution) - curr_solution)
                updated_solution = random_solution - (A*D)
                
                            
        else:
            print('EKSPLOITASI')
            D = abs(best_solution.solutions - curr_solution)
            l = np.random.uniform(-1,1)
            updated_solution = (D * (math.e ** (b*l)) * np.cos(2*math.pi*l)) + best_solution.solutions
            
        print(curr_solution)
        updated_solution = validate_range_solution(updated_solution)
        print(updated_solution)        
        updated_fitness = get_fitness(updated_solution)
        print('-error')
        print(best_solution.errors)
        print(updated_fitness.errors)
        
        # UPDATE BEST SOLUTION
        if (updated_fitness.errors < best_solution.errors):
            best_solution = updated_fitness
        
        updated_solution_list.append(updated_solution)
        
    # UPDATE ALL SOLUTION
    solutions = np.array(updated_solution_list[:]).T


plt.plot([i for i in range(len(best_solution.training_data))], best_solution.training_data, color='red', lw=1, label="Real Training Data")
plt.plot([i for i in range(len(best_solution.validation_result))], best_solution.validation_result, color='blue', lw=1, label="Prediction (Validation) Data")
plt.title('SVR-WOA', size=20)
plt.xlabel('Hari ke-', size=14)
plt.ylabel('Jumlah Kasus', size=14)
plt.legend()                 
plt.show()
    