from Functions import *



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


nol = df.shape[1]
tree = random_binary_tree(nol , random_state=0)
tree = nx.Graph(tree)

for i,j in tree.edges():
    tree[i][j]['weight'] = find_edgeweight(tree,(i,j),nol,dis)
    
dist_hat = all_tree_path_lengths(tree,nol)

rss = sum( (dist_hat[i,j]-dis[i,j])**2 for i,j in dist_hat )
rss_best=rss


tree_best = tree.copy()


for n in range(100):
    print(n)
    internal_edges = [(i,j) for i,j in tree.edges() if i>=nol and j >=nol]
    picked_edge = random.choice(internal_edges)
    NNI_info = NNI(tree,picked_edge,nol)
    edge_list_1 = list(tree.edges())
    edge_list_2 = edge_list_1.copy()
    edge_list_1.extend(NNI_info['Added_edges_1']) 
    edge_list_1 = [e for e in edge_list_1 if e not in NNI_info['Removed_edges_1'] ]
    tree_temp_1 = nx.Graph(edge_list_1)
    for (i,j) in tree_temp_1.edges():
        if (i == picked_edge[0] or i == picked_edge[1] or
                        j == picked_edge[0] or j == picked_edge[1] ):
            tree_temp_1[i][j]['weight'] = find_edgeweight (tree_temp_1,(i,j),nol,dis)
        else:
            tree_temp_1[i][j]['weight'] = tree[i][j]['weight']
    dis_hat_1 = all_tree_path_lengths(tree_temp_1 , nol)
    rss_1 = sum( (dis_hat_1[i,j]-dis[i,j])**2 for i,j in dis_hat_1 )
    if rss_1 < rss_best :
        tree_best = tree_temp_1.copy()
        rss_best = rss_1
    if rss_1 < rss:
        tree = tree_temp_1.copy()
        rss = rss_1
    else:
        edge_list_2.extend(NNI_info['Added_edges_2']) 
        edge_list_2 = [e for e in edge_list_2 if e not in NNI_info['Removed_edges_2'] ]
        tree_temp_2 = nx.Graph(edge_list_2)
        for (i,j) in tree_temp_2.edges():
            if (i == picked_edge[0] or i == picked_edge[1] or
                            j == picked_edge[0] or j == picked_edge[1] ):
                tree_temp_2[i][j]['weight'] = find_edgeweight (tree_temp_2,(i,j),nol,dis)
            else:
                tree_temp_2[i][j]['weight'] = tree[i][j]['weight']
        dis_hat_2 = all_tree_path_lengths(tree_temp_2 , nol)
        rss_2 = sum( (dis_hat_2[i,j]-dis[i,j])**2 for i,j in dis_hat_2 )
        if rss_2 < rss_best :
            tree_best = tree_temp_2.copy()
            rss_best = rss_2
        if rss_2 < rss:
            tree = tree_temp_2.copy()
            rss = rss_2
        
        
