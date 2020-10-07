from Functions import *



df=pd.read_pickle('S&P 500 time series.pkl')
df = df.sample(80, random_state=0 , axis=1)
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

random.seed()
tree_best = tree.copy()

harmonic_number = lambda n: sum(Fraction(1, d) for d in range(1, n+1)) 
local_min_stuck = float (2 * nol * harmonic_number( 2 * nol) ) #nol roughly equal to internal edges

time_passed = 5 * 60 * 60
t_end = time.time() + time_passed
counter_jump = 0

while time.time() < t_end:
    internal_edges = [(i,j) for i,j in tree.edges() if i>=nol and j >=nol]
    picked_edge = random.choice(internal_edges)
    NNI_info = NNI(tree,picked_edge,nol)
    tree_temp = nx.Graph(NNI_info[0])
    tree_temp = find_edgeweight_2(tree,tree_temp,picked_edge,nol,dis)
    rss_1 = RSS(tree_temp,nol,dis)
    if rss_1 < rss_best :
        tree_best = tree_temp.copy()
        rss_best = rss_1
    if rss_1 < rss:
        print('rss_1: %f' %(rss_1))
        tree = tree_temp.copy()
        rss = rss_1
        counter_jump=0
    else:
        tree_temp = nx.Graph(NNI_info[1])
        tree_temp = find_edgeweight_2(tree,tree_temp,picked_edge,nol,dis)
        rss_2 = RSS(tree_temp,nol,dis)
        if rss_2 < rss_best :
            print('rss_2: %f' %(rss_2))
            tree_best = tree_temp.copy()
            rss_best = rss_2
        if rss_2 < rss:
            tree = tree_temp.copy()
            rss = rss_2
            counter_jump=0
        else:
            counter_jump += 1
            if counter_jump > local_min_stuck :
                break
        
        
