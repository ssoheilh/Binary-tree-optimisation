import networkx as nx
import numpy as np
import itertools as it
from sklearn.cluster import AgglomerativeClustering as ac
import pandas as pd
import random
import operator
import math
import scipy
from scipy.cluster.hierarchy import dendrogram , linkage
import matplotlib.pyplot as plt


def hierarchy_pos(G, root, levels=None, width=1., height=1.):
    '''If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node
       levels: a dictionary
               key: level number (starting from 0)
               value: number of nodes in this level
       width: horizontal space allocated for drawing
       height: vertical space allocated for drawing'''
    TOTAL = "total"
    CURRENT = "current"
    def make_levels(levels, node=root, currentLevel=0, parent=None):
        """Compute the number of nodes for each level
        """
        if not currentLevel in levels:
            levels[currentLevel] = {TOTAL : 0, CURRENT : 0}
        levels[currentLevel][TOTAL] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                levels =  make_levels(levels, neighbor, currentLevel + 1, node)
        return levels

    def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
        dx = 1/levels[currentLevel][TOTAL]
        left = dx/2
        pos[node] = ((left + dx*levels[currentLevel][CURRENT])*width, vert_loc)
        levels[currentLevel][CURRENT] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                pos = make_pos(pos, neighbor, currentLevel + 1, node, vert_loc-vert_gap)
        return pos
    if levels is None:
        levels = make_levels({})
    else:
        levels = {l:{TOTAL: levels[l], CURRENT:0} for l in levels}
    vert_gap = height / (max([l for l in levels])+1)
    return make_pos({})


def ultrametric_distance(tree):
    
    umd_tree=np.zeros((tree.number_of_nodes(),tree.number_of_nodes()))


    for i,j in it.combinations(tree.nodes(),2):
        temp=nx.shortest_path(tree , i ,j)
        temp_dict={s:t for s,t in [((temp[s],temp[s+1]),tree[temp[s]][temp[s+1]]['weight']) for s in np.arange(len(temp)-1)] }
        umd_tree[i,j]=max(temp_dict.values())
    
    umd_tree=umd_tree+umd_tree.T
    
    return umd_tree
    

def find_weights(G,tree): #the original way of finding weights
    ''' Find the optimum weights of the edges of a spanning tree based on the edge weights of the correspoding complete graph 
        G: Complete graph
        st: spanning tree '''
    st=nx.Graph(tree)
    st_edges=list(nx.edges(st))
    
    A=np.zeros((st.number_of_nodes()-1,st.number_of_nodes()-1)) #The matrix of the number of paths an edge or two edges are on
    b=np.zeros((st.number_of_nodes()-1)) #Sum of the weight of the edge on the complete graph corresponding to the paths
    
    count=0
    for i in st_edges:
        st.remove_edge(i[0],i[1])
#        temp_0 = [node for list in nx.dfs_successors(st,i[0]).values() for node in list]+[i[0]]
#        temp_1 = [node for list in nx.dfs_successors(st,i[1]).values() for node in list]+[i[1]]
        temp=[i for i in nx.connected_components(st)]
        A[count,count]=len(temp[0])*len(temp[1]) #Getting the diagonal entries of A
#        A[count,count]=len(temp_0)*len(temp_1)
        st.add_edge(i[0],i[1])
        b[count]=sum(G[s][t]['weight'] for s in temp[0] for t in temp[1])
#        b[count]=sum(G[s][t]['weight'] for s in temp_0 for t in temp_1)
        count+=1
    #----------------------------------------------------------------------------------------
    #Getting the directed tree with the root node
        
    st_directed = nx.bfs_tree(st, 0)
    
    #----------------------------------------------------------------------------------
    #Getting the non-diagonal elements of A
    
    st_dict_2={}
    
    for i,j in it.combinations(st_directed.edges(),2):
        temp_1 = [node for list in nx.dfs_successors(st_directed,i[1]).values() for node in list]+[i[1]] 
        temp_2 = [node for list in nx.dfs_successors(st_directed,j[1]).values() for node in list]+[j[1]] 
        if set(temp_1).intersection(set(temp_2)) == set():
            st_dict_2[i,j]= len(temp_1) * len(temp_2)
        elif len(temp_1) > len(temp_2):
            st_dict_2[i,j]= (st_directed.number_of_nodes()-len(temp_1)) * len(temp_2)
        else: 
            st_dict_2[i,j]= len(temp_1) * (st_directed.number_of_nodes()-len(temp_2))
    
    
    
    edge_indices={i:j for i,j in zip(st_edges , np.arange(len(st_edges))) }
    for i,j in list(edge_indices.keys()):
        edge_indices[j,i]=edge_indices[i,j]
    
    for i,j in st_dict_2.keys():
        A[edge_indices[i],edge_indices[j]]=st_dict_2[i,j]
            
    A=np.maximum(A,A.T)
    
    #------------------------------------------------------------------------------------------
    #Solving the system of equations
#    x=[1 for i in range(len(st_edges))]
    x=scipy.linalg.cho_solve(scipy.linalg.cho_factor(A , check_finite=False) ,b , check_finite=False)   
#    x=scipy.linalg.solve(A,b)
    G_ot=nx.Graph()
    G_ot.add_weighted_edges_from([(i[0],i[1],j) for i,j in zip(st_edges,x)])
    
    return {'A' : A , 'x': x , 'b':b , 'Graph' : G_ot}

def find_weights_2(G,tree): #finding weights for one-edge and two-edge together
    st=nx.Graph(tree)
    st_edges=list(nx.edges(st))
    edge_indices={i:j for i,j in zip(st_edges , np.arange(len(st_edges))) }
    
    A=np.zeros((st.number_of_nodes()-1,st.number_of_nodes()-1)) #The matrix of the number of paths an edge or two edges are on
    b=np.zeros((st.number_of_nodes()-1)) #Sum of the weight of the edge on the complete graph corresponding to the paths
        
    #----------------------------------------------------------------------------------------
    #Getting the directed tree with the root node
    
    st_directed = nx.bfs_tree(st, 0)
    
    #----------------------------------------------------------------------------------
    #Getting the non-diagonal elements of A
    
    nodes=set(st_directed.nodes())
    number_of_nodes=len(nodes)
    
    for i,j in list(edge_indices.keys()):
        edge_indices[j,i]=edge_indices[i,j]
    
    for i,j in it.combinations_with_replacement(st_directed.edges(),2):
        if i==j:
            temp_1 = set([node for list in nx.dfs_successors(st_directed,i[1]).values() for node in list]+[i[1]])
            temp_2=nodes-temp_1
            A[edge_indices[i],edge_indices[j]]= len(temp_1) * len(temp_2)
            b[edge_indices[i]]=sum(G[s][t]['weight'] for s in temp_1 for t in temp_2)
        if i!=j:
            temp_1 = set([node for list in nx.dfs_successors(st_directed,i[1]).values() for node in list]+[i[1]])
            temp_2 = set([node for list in nx.dfs_successors(st_directed,j[1]).values() for node in list]+[j[1]]) 
            if temp_1.intersection(temp_2) == set():
                A[edge_indices[i],edge_indices[j]]= len(temp_1) * len(temp_2)
            elif len(temp_1) > len(temp_2):
                A[edge_indices[i],edge_indices[j]]= (number_of_nodes-len(temp_1)) * len(temp_2)
            elif len(temp_1) < len(temp_2): 
                A[edge_indices[i],edge_indices[j]]= len(temp_1) * (number_of_nodes-len(temp_2))
    
    A=np.maximum(A,A.T)
    #------------------------------------------------------------------------------------------
    #Solving the system of equations
    
#    x=[1 for i in range(len(st_edges))]
    x=scipy.linalg.cho_solve(scipy.linalg.cho_factor(A , check_finite=False) ,b , check_finite=False)   
#    x=scipy.linalg.solve(A,b)
    G_ot=nx.Graph()
    G_ot.add_weighted_edges_from([(i[0],i[1],j) for i,j in zip(st_edges,x)])
    
    return {'A' : A , 'x': x , 'b':b , 'Edge_indices': edge_indices , 'Graph' : G_ot}

#Finding only the rows and columns for the affected and the added edges
    
def find_weights_3(A,b,tree,G,edge_indices,removed_edge,added_edge,affected_edge): #find weights for the affected and added edges only
    #----------------------------------------------------------------------------------------
    #Getting the directed tree with the root node
    
    edge_indices_copy=edge_indices.copy()
    A_copy=np.array(A)
    b_copy=np.array(b)
    
    edge_indices_copy[(added_edge[0],added_edge[1])]=edge_indices_copy.pop((removed_edge[0],removed_edge[1]))
    edge_indices_copy[(added_edge[1],added_edge[0])]=edge_indices_copy.pop((removed_edge[1],removed_edge[0]))
    
    A_copy[edge_indices_copy[affected_edge],:]=0
    A_copy[edge_indices_copy[added_edge],:]=0
    A_copy[:,edge_indices_copy[affected_edge]]=0
    A_copy[:,edge_indices_copy[added_edge]]=0
    
    b_copy[edge_indices_copy[affected_edge]]=0
    b_copy[edge_indices_copy[added_edge]]=0
    

    st_directed = nx.bfs_tree(tree, 0)
    
    #----------------------------------------------------------------------------------
    #Getting the non-diagonal elements of A
    
    nodes=set(st_directed.nodes())
    number_of_nodes=len(nodes)
    
    for i in [k for k in [(affected_edge[0],affected_edge[1]),(added_edge[0],added_edge[1]),
                          (affected_edge[1],affected_edge[0]),(added_edge[1],added_edge[0])] 
                                        if k in st_directed.edges()]:
            for j in st_directed.edges():        
                  if i==j:
                    temp_1 = set([node for list in nx.dfs_successors(st_directed,i[1]).values() 
                        for node in list]+[i[1]])
                    temp_2=nodes-temp_1
                    A_copy[edge_indices_copy[i],edge_indices_copy[j]]= len(temp_1) * len(temp_2)
                    b_copy[edge_indices_copy[i]]=sum(G[s][t]['weight'] for s in temp_1 for t in temp_2)
                  if i!=j:
                    temp_1 = set([node for list in nx.dfs_successors(st_directed,i[1]).values() 
                            for node in list]+[i[1]])
                    temp_2 = set([node for list in nx.dfs_successors(st_directed,j[1]).values() 
                            for node in list]+[j[1]]) 
                    if temp_1.intersection(temp_2) == set():
                        A_copy[edge_indices_copy[i],edge_indices_copy[j]]= len(temp_1) * len(temp_2)
                    elif len(temp_1) > len(temp_2):
                        A_copy[edge_indices_copy[i],edge_indices_copy[j]]= (number_of_nodes-len(temp_1)) * len(temp_2)
                    elif len(temp_1) < len(temp_2): 
                        A_copy[edge_indices_copy[i],edge_indices_copy[j]]= len(temp_1) * (number_of_nodes-len(temp_2))
    
    A_copy=np.maximum(A_copy,A_copy.T)
    x=scipy.linalg.cho_solve(scipy.linalg.cho_factor(A_copy , check_finite=False) ,b_copy , check_finite=False)
    
    dict_1={j:i for i,j in edge_indices_copy.items()}
    tree_edge_list=[dict_1[i] for i in range(len(dict_1))]
    
    G_ot=nx.Graph()
    G_ot.add_weighted_edges_from([(i[0],i[1],j) for i,j in zip(tree_edge_list,x)])
    return {'A' : A_copy , 'x': x , 'b':b_copy , 'Edge_indices': edge_indices_copy , 'Graph' : G_ot}
    
#Finding only the rows and columns for the affected 

def find_weights_4(A,b,tree,G,edge_indices,removed_edge,added_edge,affected_edge): #find weights for the affected and added edges only
    #----------------------------------------------------------------------------------------
    #Getting the directed tree with the root node
    
    edge_indices_copy=edge_indices.copy()
    A_copy=np.array(A)
    b_copy=np.array(b)
    
    edge_indices_copy[(added_edge[0],added_edge[1])]=edge_indices_copy.pop((removed_edge[0],removed_edge[1]))
    edge_indices_copy[(added_edge[1],added_edge[0])]=edge_indices_copy.pop((removed_edge[1],removed_edge[0]))
    
    A_copy[edge_indices_copy[affected_edge],:]=0
    A_copy[:,edge_indices_copy[affected_edge]]=0

    
    b_copy[edge_indices_copy[affected_edge]]=0
    

    st_directed = nx.bfs_tree(tree, 0)
    
    #----------------------------------------------------------------------------------
    #Getting the non-diagonal elements of A
    
    nodes=set(st_directed.nodes())
    number_of_nodes=len(nodes)
    
    for i in [k for k in [(affected_edge[0],affected_edge[1]), (affected_edge[1],affected_edge[0])] 
                                        if k in st_directed.edges()]:
            for j in st_directed.edges():        
                  if i==j:
                    temp_1 = set([node for list in nx.dfs_successors(st_directed,i[1]).values() 
                        for node in list]+[i[1]])
                    temp_2=nodes-temp_1
                    A_copy[edge_indices_copy[i],edge_indices_copy[j]]= len(temp_1) * len(temp_2)
                    b_copy[edge_indices_copy[i]]=sum(G[s][t]['weight'] for s in temp_1 for t in temp_2)
                  if i!=j:
                    temp_1 = set([node for list in nx.dfs_successors(st_directed,i[1]).values() 
                            for node in list]+[i[1]])
                    temp_2 = set([node for list in nx.dfs_successors(st_directed,j[1]).values() 
                            for node in list]+[j[1]]) 
                    if temp_1.intersection(temp_2) == set():
                        A_copy[edge_indices_copy[i],edge_indices_copy[j]]= len(temp_1) * len(temp_2)
                    elif len(temp_1) > len(temp_2):
                        A_copy[edge_indices_copy[i],edge_indices_copy[j]]= (number_of_nodes-len(temp_1)) * len(temp_2)
                    elif len(temp_1) < len(temp_2): 
                        A_copy[edge_indices_copy[i],edge_indices_copy[j]]= len(temp_1) * (number_of_nodes-len(temp_2))
    
    A_copy=np.maximum(A_copy,A_copy.T)
    x=scipy.linalg.cho_solve(scipy.linalg.cho_factor(A_copy , check_finite=False) ,b_copy , check_finite=False)
    
    dict_1={j:i for i,j in edge_indices_copy.items()}
    tree_edge_list=[dict_1[i] for i in range(len(dict_1))]
    
    G_ot=nx.Graph()
    G_ot.add_weighted_edges_from([(i[0],i[1],j) for i,j in zip(tree_edge_list,x)])
    return {'A' : A_copy , 'x': x , 'b':b_copy , 'Edge_indices': edge_indices_copy , 'Graph' : G_ot}


def find_groups(umd,size):
    if np.min(umd)<0:
        umd_positive= umd - np.min(umd[umd<0])
        np.fill_diagonal(umd_positive , 0)
    else:
        umd_positive=umd
    clustering=ac( affinity='precomputed' , linkage='single')
    ii = it.count(umd.shape[0])
    clusters = [{'node_id': next(ii), 'left': x[0], 'right':x[1]} for x in clustering.fit(umd_positive).children_]
    
    import copy
    n_points = umd.shape[0]
    members = {i:[i] for i in range(n_points)}
    for cluster in clusters:
        node_id = cluster["node_id"]
        members[node_id] = copy.deepcopy(members[cluster["left"]])
        members[node_id].extend(copy.deepcopy(members[cluster["right"]]))
    
    on_split = pd.Series({c["node_id"]: (c["left"], c["right"]) for c in clusters})
    up_merge = {c["left"]: {"into": c["node_id"], "with": c["right"]} for c in clusters}
    up_merge.update({c["right"]: {"into": c["node_id"], "with": c["left"]} for c in clusters})
    
    groups=[]
    for i,j in [(i,j) for i,j in on_split if i<umd.shape[0] and j<umd.shape[0]]:
        temp=[i,j]
        k = on_split[on_split == (i,j)].index[0]
        sibs= [(i,j) for i,j in on_split if i==k or j==k][0]
        if sibs[0]==k:
            k_s=sibs[1]
        else:
            k_s=sibs[0]
        while k_s < umd.shape[0] and len(temp)<size:
            temp.append(k_s)
            k = on_split[on_split == (k_s,k)].index[0]
            sibs= [(i,j) for i,j in on_split if i==k or j==k][0]
            if sibs[0]==k:
                k_s=sibs[1]
            else:
                k_s=sibs[0]
        if len(temp)>size-1:
            groups.append(temp)
    return groups 


def random_tree_structure_change(tree,n_times):
    for i in np.arange(n_times):
        tree_temp_structure=list(tree.edges())
#        tree_temp_structure=[(i,k) for i,j in dict(nx.bfs_successors(tree , 0)).items() for k in j]
        random_edge=random.choice(list(tree.edges()))
        adj_nodes=set(tree.neighbors(random_edge[0])).union(set(tree.neighbors(random_edge[1])))-set([random_edge[0],random_edge[1]])
        random_adj_node=random.choice(list(adj_nodes))
        if random_adj_node in set(tree.neighbors(random_edge[0])):
            tree_temp_structure.append((random_edge[1],random_adj_node))
            tree_temp_structure=[i for i in tree_temp_structure if i not in [(random_edge[0],random_adj_node),(random_adj_node,random_edge[0])]]
        else:
            tree_temp_structure.append((random_edge[0],random_adj_node))
            tree_temp_structure=[i for i in tree_temp_structure if i not in [(random_edge[1],random_adj_node),(random_adj_node,random_edge[1])]]
        tree=nx.Graph(tree_temp_structure)
    return tree

def weighted_random_tree_structure_change(G,tree,n_times,power):
    for i in np.arange(n_times):
        tree_temp_structure=list(tree.edges())
#        tree_temp_structure=[(i,k) for i,j in dict(nx.bfs_successors(tree , 0)).items() for k in j]
        edges_sorted=sorted(tree.edges(data='weight') , key = operator.itemgetter(2))
        random_edge_weighted=edges_sorted[math.floor(random.random()**power * len(edges_sorted))]
        random_edge=(random_edge_weighted[0],random_edge_weighted[1])
        adj_nodes=set(tree.neighbors(random_edge[0])).union(set(tree.neighbors(random_edge[1])))-set([random_edge[0],random_edge[1]])
        random_adj_node=random.choice(list(adj_nodes))
        if random_adj_node in set(tree.neighbors(random_edge[0])):
            tree_temp_structure.append((random_edge[1],random_adj_node))
            tree_temp_structure=[i for i in tree_temp_structure if i not in 
                                 [(random_edge[0],random_adj_node),(random_adj_node,random_edge[0])]]
        else:
            tree_temp_structure.append((random_edge[0],random_adj_node))
            tree_temp_structure=[i for i in tree_temp_structure if i not in 
                                 [(random_edge[1],random_adj_node),(random_adj_node,random_edge[1])]]
        tree=nx.Graph(tree_temp_structure)
        A , x , b , tree = find_weights (G , tree).values()
    return tree


def find_all_groups(umd):
    all_groups={}
    for groups_size in np.arange(2,umd.shape[0]):
        temp=find_groups(umd,groups_size)
        if len(temp)!=0:
            all_groups[groups_size]=temp
        else: break
    return all_groups



def find_all_groups_and_sects(umd,stocks_inf):
    all_groups = find_all_groups(umd)
    df = pd.DataFrame(columns=['Size','Code','Sect'])
    df['Size']=all_groups.keys()
    df['Code']=df['Size'].apply(lambda x: [[stocks_inf.loc[j,'Code'] for j in i] for i in all_groups[x]])
    df['Sect']=df['Size'].apply(lambda x: [[stocks_inf.loc[j,'Sect'] for j in i] for i in all_groups[x]])
    return df



def find_groups_homogeneity(umd,stocks_inf):
    all_groups_and_sects=find_all_groups_and_sects(umd,stocks_inf)
    groups_hom={}
    for groups_size in all_groups_and_sects['Size']:
        temp=[i for i in all_groups_and_sects[all_groups_and_sects['Size']==groups_size]['Sect']][0]
        num=len([i for i in temp if len(set(i))==1])
        denum=len(temp)
        groups_hom[groups_size]="{} out of {}".format(num,denum)
    return groups_hom

def find_tree_groups_homogeneity(tree,stocks_inf):
    umd = ultrametric_distance(tree)
    all_groups_and_sects=find_all_groups_and_sects(umd,stocks_inf)
    groups_hom={}
    for groups_size in all_groups_and_sects['Size']:
        temp=[i for i in all_groups_and_sects[all_groups_and_sects['Size']==groups_size]['Sect']][0]
        num=len([i for i in temp if len(set(i))==1])
        denum=len(temp)
        groups_hom[groups_size]="{} out of {}".format(num,denum)
    return groups_hom

def draw_hierarchical_tree(tree,stocks_inf , method):
    umd=ultrametric_distance(tree)
    if np.min(umd)<0:
        umd_positive= umd - np.min(umd[umd<0])
        np.fill_diagonal(umd_positive , 0)
    else:
        umd_positive=umd
    condensed_umd=np.array([umd_positive[i,j] for i,j in dict(np.ndenumerate(umd_positive)).keys() if i<j])
    linked = linkage (condensed_umd , method=method)
    plt.figure(figsize=(10,10))
    plt.axes().spines['right'].set_visible(False)
    plt.axes().spines['top'].set_visible(False)
    plt.axes().tick_params(axis='y', which='major', labelsize=15)
#    plt.axes().tick_params(axis='x', which='major', labelsize=15)
    dendrogram (linked , orientation='top', color_threshold=0 , above_threshold_color='k' , leaf_font_size=10 , leaf_label_func=lambda x : stocks_inf.loc[x,'Sect'] )
#     plt.savefig('example_ht.png' , dpi=300)

def find_best_star(G):
    n=G.number_of_nodes()
    initial_edges=[(0,i) for i in range(1,n)]
    G_temp=nx.Graph(initial_edges)
    
    A , x , b , edge_indices , G_temp = find_weights_2 (G , G_temp).values()
    
    paths=[nx.shortest_path(G_temp,i,j) for i,j in it.combinations(G_temp.nodes(),2) ]
    z_temp = sum((sum(G_temp[path[i]][path[i+1]]['weight'] for i in np.arange(len(path)-1)) - 
                           G[path[0]][path[-1]]['weight'])**2 for path in paths )
    
    G_best=G_temp.copy()
    z_best=z_temp
    
    for i in range(1,n):
        edges=[(i,j) for j in [j for j in range(n) if j!=i]]
        G_temp=nx.Graph(edges)
        A , x , b , edge_indices , G_temp = find_weights_2 (G , G_temp).values()
    
        paths=[nx.shortest_path(G_temp,i,j) for i,j in it.combinations(G_temp.nodes(),2) ]
        z_temp = sum((sum(G_temp[path[i]][path[i+1]]['weight'] for i in np.arange(len(path)-1)) - 
                           G[path[0]][path[-1]]['weight'])**2 for path in paths )
        if z_temp<z_best:
            z_best=z_temp
            G_best=G_temp
    return {'best_star':G_best , 'z_best_star':z_best}

def random_star(G):
    random_center=random.choice(list(G.nodes()))
    n=G.number_of_nodes()
    edges=[(random_center,j) for j in [j for j in G.nodes() if j!=random_center]]
    random_star=nx.Graph(edges)
    return random_star

def draw_with_sect_labels(tree,stocks_inf):
    mapping={i:j for i,j in zip(range(stocks_inf.shape[0]),stocks_inf['Sect'])}
    tree_labelled=nx.relabel_nodes(tree, mapping)
    plt.figure(figsize=(10,10))
    nx.draw_networkx(tree_labelled)


# def draw_hierarchical_tree(tree):
#     umd=ultrametric_distance(tree)
#     if np.min(umd)<0:
#         umd_positive= umd - np.min(umd[umd<0])
#         np.fill_diagonal(umd_positive , 0)
#     else:
#         umd_positive=umd
#     condensed_umd=np.array([umd_positive[i,j] for i,j in dict(np.ndenumerate(umd_positive)).keys() if i<j])
#     linked = linkage (condensed_umd , 'single')
#     plt.figure(figsize=(10,10))
#     dendrogram (linked , orientation='top',  leaf_font_size=10)
   



#def find_weights_2(G,st):
#    ''' Find the optimum weights of the edges of a spanning tree based on the edge weights of the correspoding complete graph 
#        G: Complete graph
#        st: spanning tree '''
#    st_dict_1=nx.get_edge_attributes(st,'weight')
#    
#    A=np.zeros((st.number_of_nodes()-1,st.number_of_nodes()-1)) #The matrix of the number of paths an edge or two edges are on
#    b=np.zeros((st.number_of_nodes()-1)) #Sum of the weight of the edge on the complete graph corresponding to the paths
#    
#    count=0
#    for i,j in st_dict_1.items():
#        st.remove_edge(i[0],i[1])
#        temp=[i for i in nx.connected_components(st)]
#        A[count,count]=len(temp[0])*len(temp[1]) #Getting the diagonal entries of A
#        st.add_edge(i[0],i[1],weight=j)
#        b[count]=sum(G[s][t]['weight'] for s in temp[0] for t in temp[1])
#        count+=1
#    #----------------------------------------------------------------------------------------
#    #Getting the directed tree with the root node
#        
#    st_directed = nx.bfs_tree(st, 0)
#    
#    #----------------------------------------------------------------------------------
#    #Getting the non-diagonal elements of A
#    
#    st_dict_2={}
#    
#    for i,j in it.combinations(st_directed.edges(),2):
#        temp_1 = [node for list in nx.dfs_successors(st_directed,i[1]).values() for node in list]+[i[1]] 
#        temp_2 = [node for list in nx.dfs_successors(st_directed,j[1]).values() for node in list]+[j[1]] 
#        if set(temp_1).intersection(set(temp_2)) == set():
#            st_dict_2[i,j]= len(temp_1) * len(temp_2)
#        elif len(temp_1) > len(temp_2):
#            st_dict_2[i,j]= (st_directed.number_of_nodes()-len(temp_1)) * len(temp_2)
#        else: 
#            st_dict_2[i,j]= len(temp_1) * (st_directed.number_of_nodes()-len(temp_2))
#    
#    
#    
#    
#    edge_indices={i:j for i,j in zip(st_dict_1.keys() , np.arange(len(st_dict_1))) }
#    for i,j in list(edge_indices.keys()):
#        edge_indices[j,i]=edge_indices[i,j]
#    
#    for i,j in st_dict_2.keys():
#        A[edge_indices[i],edge_indices[j]]=st_dict_2[i,j]
#            
#    A=np.maximum(A,A.T)
#    
#    #------------------------------------------------------------------------------------------
#    #Solving the system of equations
#    
#    x=np.linalg.solve(A,b)   
#    G_ot=nx.Graph()
#    G_ot.add_weighted_edges_from([(i[0],i[1],j) for i,j in zip(st_dict_1.keys(),x)])
#    
#    return {'A' : A , 'x': x , 'b':b , 'Graph' : G_ot}
    

    
    