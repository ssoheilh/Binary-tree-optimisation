from Functions import *


time_in_seconds = 4
seed =3

df=pd.read_pickle('S&P 500 time series.pkl')
df = df.sample(30, random_state=seed , axis=1)
R=np.zeros(shape=(df.shape[0]-1,df.shape[1]))
df_values=df.values

for i in np.arange(df.shape[1]):
    for t in np.arange(1,df.shape[0]):
        R[t-1,i]=np.log(df_values[t,i]/df_values[t-1,i])

pr=np.corrcoef(R.T)
dis=np.sqrt(2*(1-pr))
np.fill_diagonal(dis , 0)


nol = df.shape[1]

def ILS_seed(seed , time_in_seconds=time_in_seconds , nol=nol , df=df , dis=dis): 
    
    random.seed()

    tree = random_binary_tree(nol)
    tree = nx.Graph(tree)
    
    file = open ('RSS_results_no_cons/Sample%d_%dmin_seed%d.txt' %(nol, time_in_seconds//60  , seed ) , 'w')
    file.close()
    

    for i,j in tree.edges():
        tree[i][j]['weight'] = find_edgeweight(tree,(i,j),nol,dis)



    rss = RSS(tree,nol,dis)
    rss_best=rss

    
    tree_best = tree.copy()

    tree_neg_NNI_best = nx.Graph()
    rss_neg_NNI_best = 100000000000000000000

    tree_pos_best = nx.Graph()
    rss_pos_best = 100000000000000000000



    harmonic_number = lambda n: sum(Fraction(1, d) for d in range(1, n+1)) 
    local_min_stuck = float (2 * nol * harmonic_number( 2 * nol) ) #nol roughly equal to internal edges

    time_passed = time_in_seconds
    t_end = time.time() + time_passed
    counter_jump = 0
    loop = 0 

    while time.time() < t_end:
        loop +=1
        internal_edges = [(i,j) for i,j in tree.edges() if i>=nol and j >=nol]
        picked_edge = random.choice(internal_edges)
        NNI_info = NNI(tree,picked_edge,nol)
        tree_temp = nx.Graph(NNI_info[0])
        tree_temp = find_edgeweight_2(tree,tree_temp,picked_edge,nol,dis)
        rss_1 = RSS(tree_temp,nol,dis)
        if rss_1 < rss_best :
            tree_best = tree_temp.copy()
            rss_best = rss_1
            file = open('RSS_results_no_cons/Sample%d_%dmin_seed%d.txt' %(nol, time_in_seconds//60  , seed ) , 'a')
            file.write('Loop {}: best total is {} \n'.format(loop , rss_best ))
            file.close()
            print('rss_1: %f' %(rss_1))
        if rss_1 < rss:
            tree = tree_temp.copy()
            rss = rss_1
            counter_jump=0
        else:
            tree_temp = nx.Graph(NNI_info[1])
            tree_temp = find_edgeweight_2(tree,tree_temp,picked_edge,nol,dis)
            rss_2 = RSS(tree_temp,nol,dis)
            if rss_2 < rss_best :
                tree_best = tree_temp.copy()
                rss_best = rss_2
                file = open('RSS_results_no_cons/Sample%d_%dmin_seed%d.txt' %(nol, time_in_seconds//60  , seed ) , 'a')
                file.write('Loop {}: best total is {} \n'.format(loop , rss_best ))
                file.close()
                print('rss_2: %f' %(rss_2))
            if rss_2 < rss:
                tree = tree_temp.copy()
                rss = rss_2
                counter_jump=0
            else:
                counter_jump += 1
                if counter_jump > local_min_stuck :
                    tree = random_binary_tree(nol)
                    tree = nx.Graph(tree)
                    for i,j in tree.edges():
                        tree[i][j]['weight'] = find_edgeweight(tree,(i,j),nol,dis)
                    rss = RSS(tree,nol,dis)
                    counter_jump =0
                    
    return rss_best


# ILS_seed(0,20)
# breakpoint()
number_of_pools=10
pool = mp.Pool(number_of_pools)
results_ILS= pool.map(ILS_seed, [i for i in number_of_pools*[seed]])
pool.close()


with open('RSS_results_no_cons/Final_Sample%d_%dmin_seed%d.txt' %(nol,time_in_seconds//60 , seed) , 'w') as file:
    for i in range(len(results_ILS)):
        file.write('%d: %f \n' %(i+1 , results_ILS[i]))

    
        