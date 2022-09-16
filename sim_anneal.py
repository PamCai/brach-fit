import numpy as np
import sys
import csv
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
import matplotlib as mpl
from brachiation_calc import rheo_calc
import collections
from random import random

####################################################
# Simulated Annealing Functions
####################################################

def sum_squared(curr_params,const_param,X_train):
    # Function that returns the sum squared errors between the predicted and experimental
    # Inputs:
    # curr_params - Dictionary of current hyperparameters
    # const_param - List of constant parameters
    # X_train - 2 rows constaining the experimental data for G1 and G2
    # Outputs:
    # SSE - sum squared of errors between the model prediction and experiment
    param_list = list(curr_params.keys())
    p0 = np.zeros(len(param_list))
    for i in range(len(param_list)):
        p0[i] = curr_params[param_list[i]]
    G11_exp = X_train[0]
    G12_exp = X_train[1]
    output1 = rheo_calc(p0[0],p0[1],p0[2],p0[3],p0[4],p0[5],const_param[0],const_param[2],const_param[1],plot_calc=False,plot_calc_all=False,print_on=False)
    SSE1 = np.sum(np.float_power(np.log(np.real(output1[1]))-np.log(G11_exp),2.)) 
    SSE2 = np.sum(np.float_power(np.log(np.imag(output1[1]))-np.log(G12_exp),2.)) 
    SSE = SSE1 + SSE2
    return SSE, output1[1]

def choose_params(param_dict, curr_params=None):
    # Function to choose parameters for next iteration
    # Inputs:
    # params_dict - Ordered dictionary of hyperparmeter search space
    # curr_params - Dictionary of current hyperparameters
    # Output:
    # next_params - Dictionary of parameters
    if curr_params:
        next_params = curr_params.copy() # makes a copy of the existing dictionary of parameters
        param_to_update = np.random.choice(list(param_dict.keys())) # randomly chooses the param to update
        param_vals = param_dict[param_to_update] # value associated with the parameter to update stored in param_dict
        print('choosing params',param_vals[0])
        next_params[param_to_update] = np.exp(np.random.normal(np.log(param_vals[0]),param_vals[1]))
    else: # first time setting params
        next_params = dict()
        for k, v in param_dict.items():
            next_params[k] = np.exp(np.random.normal(np.log(v[0]),v[1]))
    return next_params

def simul_anneal(param_dict,const_param,X_train,fn_train,maxiters=100,alpha=0.85,beta=1.3,T_0=0.4,update_iters=5):
    # Function performing hyperparameter search using simulated annealing
    # Inputs:
    # param_dict - Ordered dictionary of hyperparameter search space
    # const_param - Static parameters of the model
    # Xtrain - Train Data
    # fn_train - Function to train the model
    # maxiters - Number of iterations to perform the parameter search
    # alpha - Factor to reduce temperature
    # beta - Constant in probability estimate
    # T_0 - Initial temperature
    # update_iters - # of iterations required to update temperature
    # Output:
    # Dataframe of the parameters explored and corresponding model performance
    columns = list(param_dict.keys()) + ['Metric', 'Best Metric']
    results = pd.DataFrame(index=range(maxiters), columns=columns) # empty dataframe with columns labeled and trials labeled
    best_metric = 1.e7
    prev_metric = 1.e7
    prev_params = None
    best_params = dict() # creates empty dictionary
    all_params = dict.fromkeys(list(param_dict.keys()))
    all_params = {k: [] for k in all_params.keys()}
    T = T_0
    T_sched = [T_0]
    T_desc = int(maxiters*2.) # never drop to exponential

    for i in range(maxiters):
        print('Starting Iteration {}'.format(i))
        curr_params = choose_params(param_dict, prev_params)
        print(curr_params)
        metric, model = fn_train(curr_params, const_param, X_train)
        if metric < prev_metric:
            print('Local Improvement in metric from {:8.4f} to {:8.4f} '.format(prev_metric, metric) + ' - parameters accepted')
            prev_params = curr_params.copy()
            prev_metric = metric
            for k, v in param_dict.items():
                param_dict[k][0] = curr_params[k] # update param dictionary so that it now searches in vicinity of next best
            if metric < best_metric:
                print('Global improvement in metric from {:8.4f} to {:8.4f} '.format(best_metric, metric) + ' - best parameters updated')
                best_metric = metric
                best_params = curr_params.copy()
                best_model = model
        else:
            rnd = np.random.uniform()
            diff = metric - prev_metric
            threshold = np.exp(-beta*diff / T)
            if rnd >= threshold:
                print('No Improvement but parameters accepted. Metric change' + ': {:8.4f} threshold: {:6.4f} random number: {:6.4f}'.format(diff, threshold,rnd))
                prev_metric = metric
                prev_params = curr_params
                for k, v in param_dict.items():
                    param_dict[k][0] = curr_params[k] # update param dictionary so that it now searches in vicinity of next best
            else:
                print('No Improvement and parameters rejected. Metric change' + ': {:8.4f} threshold: {:6.4f} random number: {:6.4f}'.format(diff, threshold, rnd))
        for key, value in curr_params.items():
            all_params[key].append(value)
        results.loc[i, list(best_params.keys())] = list(best_params.values())
        results.loc[i, 'Metric'] = metric
        results.loc[i, 'Best Metric'] = best_metric
        if i % update_iters == 0:
            if i < T_desc:
                T = T_0/(1. + alpha*np.log(1+i))
            else:
                T = T*alpha
        T_sched.append(T)
    return results, best_model, all_params, T_sched, best_params

####################################################
# Loading the Data
####################################################
t = []
g1 = []
g2 = []
with open('data.csv','r') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        t.append(float(row[0]))
        g1.append(float(row[1]))
        g2.append(float(row[2]))
t = np.array(t)
g1 = np.array(g1)
g2 = np.array(g2)

####################################################
# Defining Range of Parameter Space
####################################################

param_dict = collections.OrderedDict()
param_dict['conc'] = np.array([0.4,1.e-1])
param_dict['x'] = np.array([6.e-10,1.e-1])
param_dict['ku'] = np.array([1.2,1.e-1])
param_dict['kb'] = np.array([0.03,1.e-1])
param_dict['monomers'] = np.array([565,1.e-1])
param_dict['stickers'] = np.array([360,1.e-1])
P = 500
T = 25.
const_param = [P,T,t]

X_train = []
X_train.append(g1)
X_train.append(g2)
[result, best_model, all_params, T_sched, best_params] = simul_anneal(param_dict,const_param,X_train,sum_squared,maxiters=2000,alpha=0.99,T_0=2.e3)
plt.plot(t,X_train[0],color='b',ls='',marker='.')
plt.plot(t,X_train[1],color='b',ls='',marker='.',fillstyle='none')
plt.plot(t,np.real(best_model),'r-')
plt.plot(t,np.imag(best_model),'r--')
plt.xscale('log')
plt.yscale('log')
plt.savefig('fit_test')
plt.show()
plt.close()
metric = np.array(result['Metric'])
plt.plot(range(len(metric)),metric)
#plt.savefig('metric_test')
plt.show()
plt.close()
plt.plot(range(len(T_sched)),T_sched)
plt.xscale('log')
#plt.savefig('T_sched')
plt.show()

keys, values = [], []

for key, value in param_dict.items():
    keys.append(key)
    values.append(value)       

with open('params.csv', 'w') as outfile:
    csvwriter = csv.writer(outfile)
    csvwriter.writerow(keys)
    csvwriter.writerow(values)

keys, values = [], []
for key in list(param_dict.keys()):
    keys.append(key)
    values.append(result[key])

with open('results.csv', 'w') as outfile:
    csvwriter = csv.writer(outfile)
    csvwriter.writerow(keys)
    csvwriter.writerow(values)

keys, values = [], []
for key, value in all_params.items():
    keys.append(key)
    values.append(value)
with open('all_params.csv', 'w') as outfile:
    csvwriter = csv.writer(outfile)
    csvwriter.writerow(keys)
    csvwriter.writerow(values)
    
keys, values = [], []
for key, value in best_params.items():
    keys.append(key)
    values.append(value)
with open('best_params.csv', 'w') as outfile:
    csvwriter = csv.writer(outfile)
    csvwriter.writerow(keys)
    csvwriter.writerow(values)