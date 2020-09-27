#List of functions to be used for optimisation

import networkx as nx
import numpy as np
import itertools as it
# from sklearn.cluster import AgglomerativeClustering as ac
import pandas as pd
import random
import operator
import math
import scipy
# from scipy.cluster.hierarchy import dendrogram , linkage
import matplotlib.pyplot as plt
import seaborn as sns
#import pdb
import time
import multiprocessing as mp
import pickle
# from sympy import Symbol , Matrix , solve , Eq , Sum , simplify , latex , sympify
# from sympy import *

# from gurobipy import GRB, quicksum , Model

df=pd.read_pickle('S&P 500 time series.pkl')
df_filtered = df.sample(10, random_state=0 , axis=1)

def random_binary_tree(nol): #nol: number of leaves
    edges = [(0,1)]
    leaves=[0,1]
    while len(edges) < 2 * nol - 3:
        internal=random.choice(leaves)
        edges.extend([(internal,leaves[-1]+1),(internal,leaves[-1]+2)])
        leaves.extend([leaves[-1]+1,leaves[-1]+2])
        leaves.remove(internal)
    vertices_dict = {i:j for j,i in enumerate(leaves)}
    internals = [i for i in range(2*nol-2) if i not in leaves]
    for i,j in [(i,j) for i,j in zip(internals, range(nol,2*nol-2))]:
        vertices_dict[i]=j
    leaves = {vertices_dict[i] for i in leaves}
    edges=[(vertices_dict[i],vertices_dict[j]) for i,j in edges]
    return edges #Vertices 0--nol-1 are the leaves
            
def NNI(tree,edge):
    v_i , v_j = edge[0] , edge[1]
    if (nx.degree(tree,v_i)!=3 or nx.degree(tree,v_j) != 3 ):
        print('Not an internal edge')
        raise ValueError("Not a leaf edge")
    else:
        n_i = [i for i in tree.neighbors(v_i) if i!=v_j]
        n_j = [i for i in tree.neighbors(v_j) if i!=v_i]
    edge_set_1 = [i for i in tree.edges()]+[(v_i,n_j[0]),(v_j,n_i[1])]
    edge_set_1=[e for e in edge_set_1 if e not in [(v_i,n_i[1]),(v_j,n_j[0])]]
    edge_set_2 = [i for i in tree.edges()]+[(v_i,n_j[1]),(v_j,n_i[1])]
    edge_set_2=[e for e in edge_set_2 if e not in [(v_i,n_i[1]),(v_j,n_j[1])]]    
    return edge_set_1,edge_set_2
        
       

def edge_path_func(tree):
    leaves = [i for i in tree.nodes if tree.degree(i)==1]
    leaves= sorted(leaves)
    edge_path=pd.DataFrame(columns=list(tree.edges()) , index=it.combinations(leaves,2))
    for i,j in it.combinations(leaves,2):
        temp_path=list(nx.shortest_path(tree,i,j))
        temp_path_edges=[(temp_path[i],temp_path[i+1]) for i in range(len(temp_path)-1)]
        for edge in edge_path.columns:
            if ((edge[0],edge[1]) in temp_path_edges):
                edge_path.loc[(i,j),edge]=1
            if ((edge[1],edge[0]) in temp_path_edges):
                edge_path.loc[(i,j),edge]=1
    edge_path=edge_path.fillna(0)
    return edge_path 






