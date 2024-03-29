from Functions import *



df=pd.read_pickle('S&P 500 time series.pkl')
df = df.sample(20, random_state=0 , axis=1)
R=np.zeros(shape=(df.shape[0]-1,df.shape[1]))
df_values=df.values

for i in np.arange(df.shape[1]):
    for t in np.arange(1,df.shape[0]):
        R[t-1,i]=np.log(df_values[t,i]/df_values[t-1,i])

pr=np.corrcoef(R.T)
dis=np.sqrt(2*(1-pr))
np.fill_diagonal(dis , 0)


nol = df.shape[1]


tree = random_binary_tree(nol)
tree = nx.Graph(tree)

for i,j in tree.edges():
    tree[i][j]['weight'] = find_edgeweight(tree,(i,j),nol,dis)

    

rss = RSS(tree,nol,dis)
rss_best=rss

random.seed()
tree_best = tree.copy()

tree_neg_NNI_best = nx.Graph()
rss_neg_NNI_best = 100000000000000000000

tree_pos_best = nx.Graph()
rss_pos_best = 100000000000000000000



harmonic_number = lambda n: sum(Fraction(1, d) for d in range(1, n+1)) 
local_min_stuck = float (2 * nol * harmonic_number( 2 * nol) ) #nol roughly equal to internal edges

time_passed = 5 * 60
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
                tree_neg_NNI = neg_NNI(tree_temp , nol ,dis)
                rss_neg_NNI = RSS(tree_neg_NNI , nol , dis)
                if rss_neg_NNI < rss_neg_NNI_best:
                    tree_neg_NNI_best = tree_neg_NNI.copy()
                    rss_neg_NNI_best = rss_neg_NNI
                    print('RSS_NNI_best: %f' %(rss_neg_NNI_best))
                tree_pos = zero_sub(tree_neg_NNI , nol , dis)
                rss_pos = RSS(tree_pos , nol , dis)
                if rss_pos < rss_pos_best:
                    tree_pos_best = tree_pos.copy()
                    rss_pos_best = rss_pos
                    print('RSS_pos_best: %f' %(rss_pos_best))
                tree = nx.Graph(random_binary_tree(nol))
                weighted_edges = [(i,j,find_edgeweight(tree,(i,j),nol,dis)) for  i,j in tree.edges()]
                tree.add_weighted_edges_from(weighted_edges)
                rss = RSS(tree,nol,dis)
                counter_jump =0
        
        
