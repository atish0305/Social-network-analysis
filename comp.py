# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 18:16:17 2020

@author: TUSHAR


1. Wiki
2. Enron
3. Amazon
4. Youtube
5. Enron

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
import networkx as nx

from math import log
from random import sample
import snap
from collections import Counter
import itertools

old_data = "/data/s2502585/snacs_Neigh/"
path = "/data/s2502585/snacs_Neigh/data/" 
home = "~/"

#convert(old_data, path)


ANF




def distANF(Id):
    hops  = 3
    distV = snap.TIntFltKdV()
    ## Computing neighbourhood structure for each node and dumping the list
    snap.GetAnf(net, Id.GetId(), distV, 3, False, 32)
    NS = [round(x.Dat()) for x in distV]
    return NS

def OPtoFile(filename, text):
    with open(filename, 'w') as f:
        print(text, file=f)      

def write_snap_format(G, title):
    '''Writes an edge list from the networkx graph which can be read using SNAP'''
    edge_file_name = "" + title + ".txt"
    if nx.is_directed(G):
        nx.write_edgelist(G, edge_file_name, delimiter='\t', data=False)
    else:
        f = open(edge_file_name, 'w')
        for u, v in G.edges():
            f.write('%d\t%d\n' % (u, v))
            f.write('%d\t%d\n' % (v, u))
        f.close()
        
def convert(path, data_p):
    en = pd.read_csv(str(path+'Sample_data_dip.csv')).iloc[:,[2,3]].replace('frozenset\(\{','', regex=True).replace('\}\)','', regex=True)
    en.to = en.to.str.split(', ')
    en = en.explode('to',ignore_index=True).reset_index(drop=True)
    net = nx.from_pandas_edgelist(en, 'from', 'to', create_using=nx.DiGraph())
    write_snap_format(net, str(data_p+'Enron'))
    
    um = pd.DataFrame(np.load(str(path+'user_movie.npy')), columns=['source', 'target'])
    net = nx.from_pandas_edgelist(um, 'source', 'target', create_using=nx.DiGraph())
    write_snap_format(net, str(data_p+'user-movie'))  
    
convert(old_data, path)
    

## Function definitions
def Plot(deg, cnt, ptype, title, ylab, xlab):
    plt.figure(figsize=(12,8))
    if ptype == 'bar':
        plt.bar(deg, cnt, width=0.5)
    elif ptype == 'scatter':
        plt.scatter(deg, cnt, s=5)
    plt.title(str(title))
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    plt.savefig(str(title + ptype+'.png'))
    plt.show()


class compareANFNearPy:
    def __init__(self, path, parent):
        self.data_p = path
        self.home_p = "~/"
        self.old_data = old_data
        self.dataset = list()
        self.nodes = list()
        self.edges = list()
        self.timeused_NPY = list()
        self.memory_NPY = list()
        self.timeused_ANF = list()
        self.memory_ANF = list()
    
    def EDA(self, filename):
        net = nx.read_edgelist(filename, create_using=nx.DiGraph())
        ## Network information
        edges = net.number_of_edges()
        self.edges.append(edges)
        nodes = net.number_of_nodes()
        self.nodes.append(nodes)
        OPtoFile('out.txt', str("Number of edges: %d\nNumber of nodes: %d" %(edges, nodes)))
        
        OPtoFile('out.txt', str("Density of the network is %f" % nx.density(net)))
        ## Outdegree distribution
        dg = Counter(sorted([n[1] for n in net.out_degree()], reverse=True))
        d, c = zip(*dg.items())
        Plot(d, [log(n) for n in c], 'scatter', str(filename[:-4] + "-ODD"), 'log(counts)', 'Out degree')
        
        dg = Counter(sorted([n[1] for n in net.in_degree()], reverse=True))
        d, c = zip(*dg.items())
        Plot(d, [log(n) for n in c], 'scatter', str(filename[:-4] + "-IDD"), 'log(counts)', 'In degree')
        
        ## Strongly connected components
        sc = [len(n) for n in sorted(nx.strongly_connected_components(net))]
        sc_nodes = [n for n in sorted(nx.strongly_connected_components(net))]
        sc_deg = net.degree(sc_nodes[sc.index(max(sc))])
        sc_deg = [n[1] for n in sc_deg]
        
        OPtoFile('out.txt', str("Number of strongly connected components: %d\nNodes in the largest strongly connected components: %d\nNumber of links in largest strongly connected components: %d" %(len(sc), max(sc), sum(sc_deg)/2)))
    
        ## Weakly connected components    
        wc = [len(n) for n in sorted(nx.weakly_connected_components(net))]
        wc_nodes = [n for n in sorted(nx.weakly_connected_components(net))]
        wc_deg = net.degree(wc_nodes[wc.index(max(wc))])
        wc_deg = [n[1] for n in wc_deg]
        
        OPtoFile('out.txt', str(("Number of weakly connected components: %d\nNodes in the largest weakly connected components: %d\nNumber of links in largest weakly connected components: %d" %(len(wc), max(wc), sum(wc_deg)/2))))
    
    def preprocessing(self, f_name):
        data = pd.read_csv(f_name, skiprows=4, delimiter='\t', names=['node1', 'node2']).astype(int)
        return data

    def NearPy(self, f_name):
        OPtoFile('out.txt', str('Running NearPy on '+ f_name))
        t1 = time.time()
        edge_list = self.preprocessing(f_name)
        edge_list['node2'] = pd.to_numeric(edge_list['node2'], downcast='signed')
        edge_list['node1'] = pd.to_numeric(edge_list['node1'], downcast='signed')
        data = pd.crosstab(matrix['node2'], edge_list['node1'])
        dimension = edge_list.shape[1]
        rbp = RandomBinaryProjections('rbp', 2)
        engine = Engine(dimension, lshashes=[rbp])
        for index in range(len(data)):
            v = np.array(data.iloc[index, :])
            engine.store_vector(v, 'data_%d' % index)
        current, peak = tracemalloc.get_traced_memory()
        #print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        OPtoFile('out.txt', str("Current memory usage is "+current / 10**6+"MB; Peak was "+peak / 10**6+"MB"))
        tracemalloc.stop()
        self.memory_NPY.append(peak / 10**6)
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
        #print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        #print("Total time taken is " + str(round(total_time)) + " seconds")
        OPtoFile('out.txt', str("Total time taken is " + str(round(total_time)) + " seconds"))
        tracemalloc.stop()
        self.timeused_NPY.append(round(total_time))
        
    def ANF(self, f_name):
        OPtoFile('out.txt', str('Running ANF on '+f_name))
        tracemalloc.start()
        net = snap.LoadEdgeList(snap.PNGraph, f_name, 0, 1, '\t')
        t1 = time.time()
        ## Neighborhood structure
        NS = list(map(distANF, net.Nodes()))
        t2 = time.time()
        total_time = t2-t1
        current, peak = tracemalloc.get_traced_memory()
        OPtoFile('out.txt', str(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB"))
        OPtoFile('out.txt', str("Total time taken is " + str(round(total_time)) + " seconds"))
        tracemalloc.stop()
        self.memory_ANF.append(peak / 10**6)
        self.timeused_ANF.append(round(total_time))
        with open(str(path+'NS_'+f_name), 'wb') as fp:
            pickle.dump(NS, fp)
    
    def TimePlot(self):
        plt.figure(figsize=(12,8))
        plt.plot(self.nodes, self.timeused_ANF)
        plt.plot(self.nodes, self.timeused_NPY)
        plt.legend()
        for i in range(len(self.dataset)):
            annotate(self.dataset[i][:-4], (self.nodes[i]+1000, self.timeused_ANF[i]+1000))
            annotate(self.dataset[i][:-4], (self.nodes[i]+1000, self.timeused_NPY[i])+1000)

        plt.title('ANF vs NearPy-Run time')
        plt.yscale('log')
        plt.ylabel("log scaled run time")
        plt.xlabel("Number of nodes")
        plt.savefig('ANF-NearPy-Runtime.png')
    
    def MemoryPlot(self):
        plt.figure(figsize=(12,8))
        plt.plot(self.nodes, self.memory_ANF)
        plt.plot(self.nodes, self.memory_NPY)
        plt.legend()
        for i in range(len(self.dataset)):
            annotate(self.dataset[i][:-4], (self.nodes[i]+1000, self.timeused_ANF[i]+1000))
            annotate(self.dataset[i][:-4], (self.nodes[i]+1000, self.timeused_NPY[i])+1000)

        plt.title('ANF vs NearPy-memory used')
        plt.yscale('log')
        plt.ylabel("log scaled memory used")
        plt.xlabel("Number of nodes")
        plt.savefig('ANF-NearPy-memory.png')
    
    def compare(self):
        os.chdir(self.data_p)
        directory_list = os.listdir()
        for file in directory_list:
            OPtoFile("out.txt", file)
            self.EDA(file)
            self.ANF(file)
            self.NearPy(file)
            self.dataset.append(file)
        self.TimePlot()
        self.MemoryPlot()
        os.chdir(self.home_p)
        
main = compareANFNearPy(path, old_data)
main.compare()
'''        
plot_data.append(nodes)
plot_data.append(edges)
plot_data.append(timeused)
plot_data.append(memory)

nodet = plt.plot( timeused[:5], nodes[:5], label='Nodes explored')
edget = plt.plot( timeused[:5], edges[:5], label='Edges explored')
memt = plt.plot( timeused[:5], memory[:5], label='Memory Used')
plt.legend()


plt.ylabel('log scale for nodes edges and memory')
plt.yscale('log')

plt.xlabel('Time in seconds')
plt.title('Nearpy algorithm - nodes, edges and memory in graph Vs Total time taken')
plt.savefig('usermovie node time- big.png')


plt.plot(edges, timeused)
plt.ylabel('Number of edges in a graph')
plt.xlabel('Time in seconds')
plt.title('Nearpy algorithm - nodes, edges and memory in graph Vs Total time taken')
plt.savefig('usermovie edges time.png')
'''
