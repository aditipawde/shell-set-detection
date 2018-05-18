import numpy as np
import pandas as pd
import community
import igraph as ig
import matplotlib.pyplot as plt
import ast
import re
import os
from sklearn import preprocessing as pp
import pickle

def kNN (G, vertex, k, m):
    ''' This function retains k OUTGOING nearest neighbors of vertex in G with multi edge label'''

    # Get outgoing edges 
    v_out_edges = G.es.select(_source=vertex)

    # Sort based on edge weight for each m features and keep only k members
    kNN_edge_dict = dict()
    for i in range(m):
        kNN_edge_dict[i] = sorted(v_out_edges, key=lambda e:e['weight'][i], reverse = True)[:k]

    #for i, key in enumerate(kNN_edge_dict.keys()):
        #print(key, ':', [(e.index, e['weight'][i]) for e in kNN_edge_dict[key]])

    ptr = [0] * m
    item = 0
    kNN_edges = list()
    l = min (k, len(kNN_edge_dict[0]))
    while item < l:
        # Make pointers point to all new items
        for i, p in enumerate(ptr):
            while p < len (kNN_edge_dict[i]) and kNN_edge_dict[i][p] in kNN_edges:
                p += 1
                ptr[i] = p
        # Get all top items
        all_tops = np.array([kNN_edge_dict[i][p]['weight'][i] for i,p in enumerate(ptr)])

        # Select maximum
        max_ind = np.argwhere(all_tops == np.max(all_tops)).T[0]
        sel_ind = np.random.choice(max_ind) if len(max_ind) > 1 else max_ind[0]
        #print('All tops: ', all_tops, ' Sel ind: ', sel_ind)

        # Add that edge to kNN_edges, replace edge weight with max of edge label
        sel_edge = kNN_edge_dict[sel_ind][ptr[sel_ind]]
        G.es[sel_edge.index]['weight'] = max(G.es[sel_edge.index]['weight'])
        kNN_edges.append(sel_edge)
        item += 1
        ptr[sel_ind] += 1

    # Delete remaining edges
    e_to_be_removed = [e.index for e in v_out_edges if e not in kNN_edges]

    # Return kNN vertices
    v_kNN = [e.target for e in kNN_edges]
    kNN_wt = [e['weight'] for e in kNN_edges]
    #print(vertex, ':', list(zip(v_kNN, kNN_wt)))

    return v_kNN, e_to_be_removed

def mnv_clusters (c1, c2, kNN_dict):
    total_mnv = 0
    for u in c1:
        for v in c2:
            total_mnv += mnv_vertices(u,v,kNN_dict)
    return total_mnv / (len(c1) * len(c2))

def mnv_vertices (u, v, kNN_dict):
    
    if u != v:
        rank_u = kNN_dict[v].index(u) if u in kNN_dict[v] else 500.0
        rank_v = kNN_dict[u].index(v) if v in kNN_dict[u] else 500.0
        return rank_u + rank_v
    else:
        return 800.0

def mutual_NN (G, k, c, kNN_dict, mnv_mat_org):
    clusters = [{v} for v in G_multi.vs.indices]

    mnv_mat =  mnv_mat_org
    max_mnv = 9*k+50
    #max_mnv = 2*k+20
    while len(clusters) > c:
        N = len(clusters)
        min_mnv = float('inf')
        for row in range(1, N):
            for col in range(row):
                
                if mnv_mat[row][col] < min_mnv:
                    min_mnv = mnv_mat[row][col]
                    i = col
                    j = row

        print('(i,j):', i, j, 'mnv:', min_mnv)
        if min_mnv > max_mnv:
            break
        #merge j into i
        c_i, c_j = clusters[i], clusters[j]
        print('c_i:', c_i, 'c_j:', c_j)
        clusters[i] = set.union(c_i, c_j)
        clusters.pop(j)
        print('merged:', clusters[i]) 
        
        mnv_mat = np.delete(mnv_mat, [j], axis = 1)
        mnv_mat = np.delete(mnv_mat, [j], axis = 0)
        print('Shape:', mnv_mat.shape)
        
        for l in range(N-1):
            if i != l:
                mnv_mat[i][l] = mnv_clusters (clusters[i], clusters[l], kNN_dict)
            else:
                mnv_mat[i][l] = 500.0

    return clusters

k_list = [4,5,6,7,8,9,10,11,12]; m = 2; c = 1
F = ['n_tran', 'mean_amount']
'''
for f in F:
    if np.max(e_set[f]) - np.min(e_set[f]) > 10 ** 5:
        e_set[f] = pp.scale(np.log(e_set[f])) + 10
    else:
        e_set[f] = pp.scale(e_set[f]) + 10
'''        
results = list()


for k in k_list:
    print('k=',k)
    '''
    G_multi = ig.Graph(directed=True)
    G_multi.add_vertices(v_list)
    G_multi.add_edges(e_list)
    G_multi.vs['label'] = v_list
    e_multi = [tuple(x) for x in e_set[F].values]
    G_multi.es['weight'] = e_multi
    G_multi.es['label'] = e_multi
    '''
    graph_name = r'G:\Study\2017-2018\Work\Experimentation\Results_10_5\MultiEdge\mean_amt_n_tran' + '\graph_' + str(k) + '_' + 'NN.pickle'
    kNN_dict_file_name = r'G:\Study\2017-2018\Work\Experimentation\Results_10_5\MultiEdge\mean_amt_n_tran' + '\\' + str(k) + 'NN_dict.pickle'

    
    '''
    kNN_dict = dict()
    e_remove = dict()
    for v in G_multi.vs.indices:
        kNN_dict[v], e_remove[v] = kNN(G_multi, v, k, m)
        G_multi.delete_edges(e_remove[v])
        
    G_multi.save (graph_name)
    '''
    with open(graph_name, 'rb') as f:
        G_multi = pickle.load(f)
        
    with open(kNN_dict_file_name, 'rb') as f:
        kNN_dict = pickle.load(f)
        
    N =  len(G_multi.vs)

    mnv_mat_org = np.ones((N,N), dtype=float)
    mnv_mat_org = 200 * mnv_mat_org

    for u in G_multi.vs.indices:
        for v in G_multi.vs.indices:
            if u in kNN_dict[v] or v in kNN_dict[u]:
                mnv_mat_org[u][v] = mnv_vertices(u,v,kNN_dict)  
                mnv_mat_org[v][u] = mnv_mat_org[u][v]
                #print('mnv', u, v, ':', mnv_mat_org[u][v])
    print('And done!!')
    cls = mutual_NN (G_multi, k, c, kNN_dict, mnv_mat_org)
    cls_size_filter = [c for c in cls if len(c) > 2]
    results.append((k, cls_size_filter))

exp_res_path = r'G:\Study\2017-2018\Work\Experimentation\Results_10_5\MultiEdge\MutualNN\exp_mean_amt_n_tran.csv'
exp_res_df = pd.DataFrame(data= results, columns = ['k', 'sets'])
exp_res_df.to_csv(exp_res_path)
