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
from fractions import Fraction 
# from sympy import Symbol , Matrix , solve , Eq , Sum , simplify , latex , sympify
# from sympy import *

# from gurobipy import GRB, quicksum , Model

df=pd.read_pickle('S&P 500 time series.pkl')
df = df.sample(100, random_state=0 , axis=1)
R=np.zeros(shape=(df.shape[0]-1,df.shape[1]))
df_values=df.values

for i in np.arange(df.shape[1]):
    for t in np.arange(1,df.shape[0]):
        R[t-1,i]=np.log(df_values[t,i]/df_values[t-1,i])

pr=np.corrcoef(R.T)
dis=np.sqrt(2*(1-pr))
np.fill_diagonal(dis , 0)


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
            
def NNI(tree,edge,nol):
    v_i , v_j = edge[0] , edge[1]
    if (v_i < nol or v_j < nol):
        raise ValueError("Not a leaf edge")
    else:
        n_i = [i for i in tree.neighbors(v_i) if i!=v_j]
        n_j = [i for i in tree.neighbors(v_j) if i!=v_i]
    edge_set_1 = [i for i in tree.edges()]+[(v_i,n_j[0]),(v_j,n_i[1])]
    edge_set_1=[e for e in edge_set_1 if e not in [(v_i,n_i[1]),(v_j,n_j[0])]]
    edge_set_2 = [i for i in tree.edges()]+[(v_i,n_j[1]),(v_j,n_i[1])]
    edge_set_2=[e for e in edge_set_2 if e not in [(v_i,n_i[1]),(v_j,n_j[1])]]    
    return edge_set_1,edge_set_2
   
def lambda_edgeweight (tree,edge,nol):
    v_i , v_j = edge[0] , edge[1]
    tree_directed = nx.dfs_tree(tree,v_i)
    if (v_i >= nol and v_j >= nol): #Checking internal edge
        n_i = [k for k in tree_directed.neighbors(v_i) if k!=v_j]
        v_A , v_B = n_i[0] , n_i[1]
        leaves_A=[v for v in list(nx.descendants(tree_directed,v_A))+[v_A] if v<nol]
        leaves_B=[v for v in list(nx.descendants(tree_directed,v_B))+[v_B] if v<nol]
        n_j = [k for k in tree_directed.neighbors(v_j)]
        v_C , v_D = n_j[0] , n_j[1]
        leaves_C=[v for v in list(nx.descendants(tree_directed,v_C))+[v_C] if v<nol]
        leaves_D=[v for v in list(nx.descendants(tree_directed,v_D))+[v_D] if v<nol]
        return {'Lambda': Fraction(len(leaves_A)*len(leaves_D)+len(leaves_B)*len(leaves_C) ,
                        (len(leaves_A)+len(leaves_B))*(len(leaves_C)+len(leaves_D))) ,
                'A' : leaves_A ,'B' : leaves_B , 'C' : leaves_C , 'D' : leaves_D }
    else:
        raise ValueError("Not an internal edge")


def average_distance (leave_set_1,leave_set_2,dis_matrix):
    dis_matrix_filtered = dis[np.ix_(leave_set_1,leave_set_2)]
    return np.sum(dis_matrix_filtered)/(len(leave_set_1)*len(leave_set_2))
    


def find_edgeweight (tree,edge,nol,dis_matrix):
    if (edge not in tree.edges):
        raise ValueError("Edge does not exist")
    v_i , v_j = edge[0] , edge[1]
    if (v_i >= nol and v_j >= nol):
        alpha , A , B , C , D = lambda_edgeweight(tree, edge, nol).values()
        d_AC , d_BD , d_AD , d_BC , d_AB , d_CD = (
            average_distance(A,C,dis_matrix) , average_distance(B,D,dis_matrix) ,
            average_distance(A,D,dis_matrix) , average_distance(B,C,dis_matrix) ,
            average_distance(A,B,dis_matrix) , average_distance(C,D,dis_matrix) )
        return (Fraction(1,2) * (alpha * (d_AC + d_BD) + (1-alpha) * (d_AD + d_BC)
                                - (d_AB + d_CD) ) )
    elif (v_i < nol or v_j < nol):
        minimum , maximum = min(v_i,v_j) , max(v_i,v_j)
        v_i , v_j = minimum , maximum
        tree_directed = nx.dfs_tree(tree,v_i)
        n_j = [k for k in tree_directed.neighbors(v_j)]
        v_A , v_B = n_j[0] , n_j[1]
        leaves_A=[v for v in list(nx.descendants(tree_directed,v_A))+[v_A] if v<nol]
        leaves_B=[v for v in list(nx.descendants(tree_directed,v_B))+[v_B] if v<nol]
        return  ( Fraction(1,2)*(average_distance(leaves_A,[v_i],dis_matrix) +
                              average_distance(leaves_B,[v_i],dis_matrix) -
                              average_distance(leaves_A,leaves_B,dis_matrix) ) )
    
       

def edge_path_func(tree):
    # breakpoint()
    leaves = [i for i in tree.nodes if tree.degree(i)==1]
    leaves= sorted(leaves)
    edge_path=pd.DataFrame(columns=list(tree.edges()) , index=it.combinations(leaves,2))
    for i,j in it.combinations(leaves,2):
        temp_path=list(nx.shortest_path(tree,i,j))
        temp_path_edges=[(temp_path[i],temp_path[i+1]) for i in range(len(temp_path)-1)]
        for edge in edge_path.columns:
            if ((edge[0],edge[1]) in temp_path_edges):
                edge_path.loc[[(i,j)],[edge]]=1
            if ((edge[1],edge[0]) in temp_path_edges):
                edge_path.loc[[(i,j)],[edge]]=1
    edge_path=edge_path.fillna(0)
    return edge_path 






