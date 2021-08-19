# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 09:18:00 2021

@author: MUZANNI
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from random import random

NUMBER_OF_SOLUTION = 40
NUMBER_OF_ITERATIONS = 20
THRESHOLD_UPDATE_PROBABILITY = 0.5
b = 2


fileName = 'data/regression.xlsx'
rawFile = pd.read_excel(fileName)

inputs = np.array([(i, 1) for i in rawFile['X']]).reshape(-1,2)
outputs = rawFile['Y'].to_numpy()

# Get Fitness of list of solution / agent
def get_fitness(solutions):    
    outputs_predict = np.dot(inputs, solutions)
    errors = np.array([(i - j) ** 2 for i, j in zip(outputs, outputs_predict)]).sum() / len(outputs)
    return {
        'errors': errors,
        'solutions': solutions
        }

def get_random(multiplier = 1):
    return random()*multiplier

def get_best_solution(solutions):
    fitness_list = [get_fitness(solutions[:,idx]) for idx in range(len(solutions[0]))]
    sorted_fitness_list = sorted(fitness_list, key=lambda fitness: fitness['errors'])
    return sorted_fitness_list[0]

# Generate Random Solution
solutions = np.array([[get_random(100), get_random(100)] for i in range(NUMBER_OF_SOLUTION)]).transpose()

# 
fitness = get_fitness(solutions)

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
                D = abs((C*best_solution['solutions']) - curr_solution)
                updated_solution = best_solution['solutions'] - (A*D)
                
                
            else:
                print('SEARCHING')
                random_solution = solutions[:, int(random()*len(solutions))]
                D = abs((C*random_solution) - curr_solution)
                updated_solution = random_solution - (A*D)
                
                            
        else:
            print('EKSPLOITASI')
            D = abs(best_solution['solutions'] - curr_solution)
            l = np.random.uniform(-1,1)
            updated_solution = (D * (math.e ** (b*l)) * np.cos(2*math.pi*l)) + best_solution['solutions']
            
        print(curr_solution)
        print(updated_solution)
        updated_fitness = get_fitness(updated_solution)
        print('-error')
        print(best_solution['errors'])
        print(updated_fitness['errors'])
        
        # UPDATE BEST SOLUTION
        if (updated_fitness['errors'] < best_solution['errors']):
            best_solution = updated_fitness
        
        updated_solution_list.append(updated_solution)
        
    # UPDATE ALL SOLUTION
    solutions = np.array(updated_solution_list[:]).T

prediction = np.dot(inputs, best_solution['solutions'])

plt.scatter(rawFile.loc[rawFile['Tipe'] == 0]['X'], rawFile.loc[rawFile['Tipe'] == 0]['Y'], s=100, c='blue', alpha=0.65, label='Tipe A')   # plot scatter
plt.scatter(rawFile.loc[rawFile['Tipe'] == 1]['X'], rawFile.loc[rawFile['Tipe'] == 1]['Y'], s=100, c='red', alpha=0.65, label='Tipe B')   # plot scatter
plt.plot(rawFile['X'], prediction, color='#000000', lw=3, label=f"GA Result, m = {best_solution['solutions'][0]:.2f}, c = {best_solution['solutions'][1]:.2f}")
plt.title('Metode WOA', size=20)
plt.xlabel('X', size=14)
plt.ylabel('Y', size=14)
plt.legend()                 
plt.show()
    



