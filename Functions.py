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
import time
import multiprocessing as mp
import pickle
from fractions import Fraction 
# from sympy import Symbol , Matrix , solve , Eq , Sum , simplify , latex , sympify
# from sympy import *

# from gurobipy import GRB, quicksum , Model

# df=pd.read_pickle('S&P 500 time series.pkl')
# df = df.sample(100, random_state=0 , axis=1)
# R=np.zeros(shape=(df.shape[0]-1,df.shape[1]))
# df_values=df.values

# for i in np.arange(df.shape[1]):
#     for t in np.arange(1,df.shape[0]):
#         R[t-1,i]=np.log(df_values[t,i]/df_values[t-1,i])

# pr=np.corrcoef(R.T)
# dis=np.sqrt(2*(1-pr))
# np.fill_diagonal(dis , 0)


def random_binary_tree(nol , random_state='None'): #nol: number of leaves
    if random_state!='None':
        random.seed(random_state)
    else:
        random.seed()
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
    edge_list = list(tree.edges())
    if (v_i < nol or v_j < nol):
        raise ValueError("Not a leaf edge")
    else:
        n_i = [i for i in tree.neighbors(v_i) if i!=v_j]
        n_j = [i for i in tree.neighbors(v_j) if i!=v_i]
    edge_list_1 = [i for i in edge_list]+[(v_i,n_j[0]),(v_j,n_i[1])]
    edge_list_1=[e for e in edge_list_1 if e not in [(v_i,n_i[1]),(v_j,n_j[0]),(n_i[1],v_i),(n_j[0],v_j)]]
    edge_list_2 = [i for i in edge_list]+[(v_i,n_j[1]),(v_j,n_i[1])]
    edge_list_2=[e for e in edge_list_2 if e not in [(v_i,n_i[1]),(v_j,n_j[1]),(n_i[1],v_i),(n_j[1],v_j)]]    
    return ( edge_list_1 , edge_list_2 )
   
def NNI_n(tree , nol , n):
    for i in range(n):
        internal_edges = [(i,j) for i,j in tree.edges() if i>=nol and j >=nol]
        random_edge = random.choice(internal_edges)
        candidate_edge_lists = NNI(tree, random_edge, nol)
        tree = nx.Graph(random.choice(candidate_edge_lists))
    return tree
    
    
    
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
    dis_matrix_filtered = dis_matrix[np.ix_(leave_set_1,leave_set_2)]
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
    
def find_edgeweight_2(tree,tree_temp,picked_edge,nol,dis): #Finds all edges weights after NNI
    for (i,j) in tree_temp.edges():
            if (i == picked_edge[0] or i == picked_edge[1] or
                            j == picked_edge[0] or j == picked_edge[1] ):
                tree_temp[i][j]['weight'] = find_edgeweight (tree_temp,(i,j),nol,dis)
            else:
                tree_temp[i][j]['weight'] = tree[i][j]['weight']
    return tree_temp

def find_edgeweight_3(tree,edge,nol,dis): #Finds only the 5 edges weights after NNI
    edges=neighbor_edges(tree, edge) +[edge]
    edge_weights={e:find_edgeweight (tree,e,nol,dis) for e in edges }
    return edge_weights



def neighbor_edges(tree,edge):
        i,j = edge[0] , edge[1]
        l_1 = [(k,i) for k in tree.neighbors(i) if k != j]
        l_2 = [(j,k) for k in tree.neighbors(j) if k != i]
        return  l_1 + l_2


def all_tree_path_lengths(tree,nol):
    dis_hat = {}
    root_paths={}
    for i in range(1,nol):
        root_paths[i]=path_edges(tree,0,i)
        dis_hat[0,i] = sum(tree[e[0]][e[1]]['weight'] for e in root_paths[i])          
    # breakpoint()
    for i in range(1,nol-1):
        for j in range(i+1,nol): 
            common_path=[]
            minimum = min(len(root_paths[i]) , len(root_paths[j]))
            for k in range(minimum):
                if (root_paths[i][k] == root_paths[j][k]):
                    common_path.append(root_paths[i][k])
                else:
                    break
            dis_hat[i,j] = dis_hat[0,i] + dis_hat[0,j] - 2 * (
                sum(tree[e[0]][e[1]]['weight'] for e in common_path))
    return dis_hat
            

def tree_path_lengths(tree,path,nol):
    i , j = min(path[0],path[1]) , max(path[0],path[1])
    dis_hat = {}
    root_paths={}
    for i in range(1,nol):
        root_paths[i]=path_edges(tree,0,i)
        dis_hat[0,i] = sum(tree[e[0]][e[1]]['weight'] for e in root_paths[i])          
    # breakpoint()
    for i in range(1,nol-1):
        for j in range(i+1,nol): 
            common_path=[]
            minimum = min(len(root_paths[i]) , len(root_paths[j]))
            for k in range(minimum):
                if (root_paths[i][k] == root_paths[j][k]):
                    common_path.append(root_paths[i][k])
                else:
                    break
            dis_hat[i,j] = dis_hat[0,i] + dis_hat[0,j] - 2 * (
                sum(tree[e[0]][e[1]]['weight'] for e in common_path))
    return dis_hat

               

def path_edges(tree,root,leaf):
    predecessors = nx.dfs_predecessors(tree,root)
    parent = predecessors[leaf]
    child=leaf
    path_edge_list=[(parent,child)]
    while parent!=root:
        temp=predecessors[parent]
        child=parent
        parent=temp
        path_edge_list = [(parent,child)] + path_edge_list
    return path_edge_list





def neg_NNI(tree,nol,dis):
    # breakpoint()
    negative_edges={(i,j):k for i,j,k in tree.edges(data='weight') if k<0}
    # sum_neg=sum(negative_edges.values())
    flag=0
    count=0
    while flag==0:
        count +=1
        flag=1
        for e in negative_edges:
            NNI_info = NNI(tree,e,nol)
            edges_5 = {(i,j):tree[i][j]['weight'] for i,j in neighbor_edges(tree,e) +[e]}
            sum_neg_5 = sum(i for i in edges_5.values() if i<0)
            for NNI_index in range(2):
                tree_temp = nx.Graph(NNI_info[NNI_index])
                edges_5_temp=find_edgeweight_3(tree_temp,e,nol,dis) 
                # breakpoint()
                sum_neg_5_temp = sum(i for i in edges_5_temp.values() if i<0)
                edges_5_temp.update({(e[1],e[0]):k for e,k in edges_5_temp.items()})
                # sum_neg_5 = sum([k for i,j,k in tree.edges(data='weight') if k<0])
                if sum_neg_5_temp > sum_neg_5:
                    for i,j in tree_temp.edges():
                        tree_temp[i][j]['weight'] = edges_5_temp[i,j] if (i,j) in edges_5_temp else tree[i][j]['weight']
                        # sum_neg=sum(negative_edges.values())
                    # breakpoint()
                    tree=tree_temp.copy()
                    flag=0
                    break
            # breakpoint()
            if flag==0:
                negative_edges={(i,j):k for i,j,k in tree.edges(data='weight') if k<0}
                # sum_neg=sum(negative_edges.values())
                break
    # breakpoint()
    return tree 



def zero_sub(tree,nol,dis):
    # breakpoint()
    A , b , edge_indices = normal_eq(tree , nol , dis).values()
    w = {(i,j):tree[i][j]['weight'] for i,j in edge_indices.keys()}
    negative_edges={(i,j):k for (i,j),k in w.items() if k<0 }
    sum_neg=sum(negative_edges.values())
    flag=0
    count=0
    irows , icols = np.indices(( 2*nol-3 , 2*nol-3 ))
    while flag==0:
        flag=1
        # breakpoint()
        for (i,j) in negative_edges:
            w[i,j]  = 0
            mask = (irows!=edge_indices[i,j])&(icols!=edge_indices[i,j])
            A_temp = A[mask].reshape(len(irows)-1 , len(icols)-1)
            b_temp = b[np.arange(2*nol-3) != edge_indices[i,j]]
            w_temp = scipy.linalg.cho_solve(scipy.linalg.cho_factor(
                    A_temp , check_finite=False) ,b_temp , check_finite=False) 
            sum_neg_temp = sum(w_temp[w_temp<0])
            if sum_neg_temp > sum_neg:
                count +=1
                flag = 0 
                w = {i:j for i,j in zip(edge_indices.keys(),np.insert(w_temp, edge_indices[i,j] , 0))}
                negative_edges={(i,j):k for (i,j),k in w.items() if k<0 }
                sum_neg = sum_neg_temp
                break
    # breakpoint()
    tree_out = nx.Graph()
    tree_out.add_weighted_edges_from([(i,j,k) for (i,j),k in w.items()])
    return tree_out
  


def normal_eq(tree,nol,dis): #Build the matrix and vector for normal equations
    tree_directed = nx.dfs_tree(tree , 0)
    leaves = set(np.arange(nol))
    edge_indices = {i:j for i,j in zip(tree_directed.edges() , np.arange(2*nol-3))}
    A = np.zeros((2*nol-3 , 2*nol-3))
    b = np.zeros((2*nol-3))
    
    # for i,j in list(edge_indices.keys()):
    #     edge_indices[j,i]=edge_indices[i,j]
    
    for i,j in it.combinations_with_replacement(tree_directed.edges(),2):
        if i==j:
            temp_1 = {v for v in list(nx.descendants(tree_directed,i[1]))+[i[1]] if v < nol}
            temp_2=leaves-temp_1
            A[edge_indices[i],edge_indices[j]]= len(temp_1) * len(temp_2)
            b[edge_indices[i]]=sum(dis[s][t] for s in temp_1 for t in temp_2)
        if i!=j:
            temp_1 = {v for v in list(nx.descendants(tree_directed,i[1]))+[i[1]] if v < nol}
            temp_2 = {v for v in list(nx.descendants(tree_directed,j[1]))+[j[1]] if v < nol}
            if temp_1.intersection(temp_2) == set():
                A[edge_indices[i],edge_indices[j]]= len(temp_1) * len(temp_2)
            elif len(temp_1) > len(temp_2):
                A[edge_indices[i],edge_indices[j]]= (len(leaves)-len(temp_1)) * len(temp_2)
            elif len(temp_1) < len(temp_2): 
                A[edge_indices[i],edge_indices[j]]= len(temp_1) * (len(leaves)-len(temp_2))
    
    A=np.maximum(A,A.T)
    return {'A' : A  , 'b':b , 'Edge_indices': edge_indices }



def negative_edges(tree):
    return {(i,j):k for i,j,k in tree.edges(data='weight') if k<0}

def zero_edges(tree):
    return {(i,j):k for i,j,k in tree.edges(data='weight') if k==0}


def RSS(tree,nol,dis):
    dis_hat = all_tree_path_lengths(tree , nol)
    return sum( (dis_hat[i,j]-dis[i,j])**2 for i,j in dis_hat )

def zero_replacement(tree):
    tree_out=nx.Graph()
    tree_out.add_edges_from(tree.edges())
    for i,j,k in tree.edges(data='weight'):
        if k>0:
            tree_out[i][j]['weight']=tree[i][j]['weight']
        else:
            tree_out[i][j]['weight']=0
    return tree_out
    


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







