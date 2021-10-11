# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:33:52 2020

@author: atish
"""

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.filters import NearestFilter, UniqueFilter
import numpy as np
import os
import pandas as pd
import time
import tracemalloc
import matplotlib.pyplot as plt
nodes = []
edges = []
timeused = []
memory = []

import os
os.chdir('C:/Users/atish/Downloads/Assignments/Assignments' +\
         '/social network analysis/project')


directory_list = os.listdir()

plot_data = []
for file in directory_list:
    
    
    matrix = pd.read_csv(directory_list[1])
    n = [round(len(matrix)/100), round(len(matrix)/50), round(len(matrix)/25),
         round(len(matrix)/5)]
        
    for number in n:
        matrix = pd.read_csv(directory_list[0], nrows=number)
    
        matrix.columns = ['node1', 'node2']
        matrix = matrix.astype(int)
        t1 = time.time()
        tracemalloc.start()
        #matrix = matrix.iloc[:number, :]
        
        val = matrix.drop_duplicates().node2.value_counts()
        val = val.sort_index()
    
    
        matrix['node2'] = pd.to_numeric(matrix['node2'], downcast='signed')
        matrix['node1'] = pd.to_numeric(matrix['node1'], downcast='signed')
        data = pd.crosstab(matrix['node2'], matrix['node1'])
        dimension = data.shape[1]
        rbp = RandomBinaryProjections('rbp', 2)
        engine = Engine(dimension, lshashes=[rbp])
        nodes.append(len(data))
        edges.append(len(matrix))
        
        for index in range(len(data)):
            v = np.array(data.iloc[index, :])
            engine.store_vector(v, 'data_%d' % index)
        
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        tracemalloc.stop()
        memory.append(peak / 10**6)
        tracemalloc.start()
        
        query_data = data.index
        
        
        N = []
        for i in range(len(query_data)):
            query = data[data.index == query_data[i]].iloc[0, :]
            query = pd.to_numeric(query, downcast='integer')
            query = np.array(query)
            N.append(engine.neighbours(query))
        
        t2 = time.time()
        
        total_time = t2-t1
        
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        print("Total time taken is " + str(round(total_time)) + " seconds")
        tracemalloc.stop()
        
        
        
        timeused.append(round(total_time))
        
    


nodet = plt.plot( timeused[4:], nodes[4:], label='Nodes explored')
edget = plt.plot( timeused[4:], edges[4:], label='Edges explored')
memt = plt.plot( timeused[4:], memory[4:], label='Memory Used')
plt.legend()


plt.ylabel('log scale for nodes edges and memory')
plt.yscale('log')



plt.xlabel('Time in seconds')
plt.title('Nearpy algorithm - nodes, edges and memory in graph Vs Total time taken')
plt.savefig('enron.png')


plt.plot(edges, timeused)
plt.ylabel('Number of edges in a graph')
plt.xlabel('Time in seconds')
plt.title('Nearpy algorithm - nodes, edges and memory in graph Vs Total time taken')
plt.savefig('usermovie edges time.png')





Dataset = matrix
a = matrix['node1'].unique()

Dataset['node2'] = Dataset['node2'].astype(int)
Dataset['node1'] = Dataset['node1'].str.replace('_', '')
from keras.preprocessing.text import one_hot
matrix_one_hot = []

for text in (Dataset['node1']):
    try:
        matrix_one_hot.append(one_hot(str(text),len(a))[0])
    except:
        matrix_one_hot.append(0)
  
matrix['node1'] = Dataset['node1']
matrix['node2'] = Dataset['node2']
  
matrix.to_csv('large_noweights.csv', index=False)

matrix.columns = matrix.columns.str.replace('^ +| +$', '_')
matrix.columns = matrix.columns.str.replace('^ +', '_')
matrix.columns = matrix.columns.str.replace(' +$', '_')
matrix['node2'] = matrix['node2'].astype(str)

matrix['node2'] = matrix['node2'].replace('28487, 11896, 27217', '28487')


Data = pd.read_csv('enron_gephi.csv')













