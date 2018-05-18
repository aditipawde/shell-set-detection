import igraph as ig
import pandas as pd
import numpy as np
import ast
import re
from sklearn.preprocessing import scale
from math import log
import pickle

# kNN_single
def kNN_single_edge (G, vertex, k):
    ''' This function retains k OUTGOING nearest neighbors of vertex in G'''
    # Get outgoing edges 
    v_out_edges = G.es.select(_source=vertex)

    # Sort based on edge weight
    v_out_edges_sorted = sorted(v_out_edges, key=lambda e:e['weight'], reverse = True)

    # Retain k edges
    v_out_edges_sorted_k = v_out_edges_sorted[:k]

    # Delete remaining edges
    e_to_be_removed = [e.index for e in v_out_edges if e not in v_out_edges_sorted_k]

    # Return kNN vertices
    v_kNN = [e.target for e in v_out_edges_sorted_k]

    return v_kNN, e_to_be_removed

# m_way_merge
def kNN_m_way (G, vertex, k, m, is_local_std):
    ''' This function retains k OUTGOING nearest neighbors of vertex in G with multi edge label'''

    # Get outgoing edges 
    v_out_edges = G.es.select(_source=vertex)

    # Local standardization
    if is_local_std and len(v_out_edges) > 0:
        v_out_edges['weight'] = [tuple(b) for b in scale(v_out_edges['weight']) + 10]

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

# AND all kNN lists
def kNN_AND (G, vertex, k, m, is_local_std):
    ''' This function retains k OUTGOING nearest neighbors of vertex in G'''
    # Get outgoing edges 
    v_out_edges = G.es.select(_source=vertex)

    # Sort based on edge weight for each m features and keep only k members
    kNN_edge_dict = dict()
    for i in range(m):
        kNN_edge_dict[i] = sorted(v_out_edges, key=lambda e:e['weight'][i], reverse = True)[:k]

    intersecting_edges = set.intersection(*[set(val) for val in kNN_edge_dict.values()])

    # Retain k edges
    v_out_edges_sorted_k = list(intersecting_edges)[:k]

    # Delete remaining edges
    e_to_be_removed = [e.index for e in v_out_edges if e not in v_out_edges_sorted_k]

    # Return kNN vertices
    v_kNN = [e.target for e in v_out_edges_sorted_k]

    return v_kNN, e_to_be_removed

# kNN m avg
def kNN_m_avg (G, vertex, k, m, is_local_std):
    ''' This function retains k OUTGOING nearest neighbors of vertex in G with multi edge label'''

    # Get outgoing edges 
    v_out_edges = G.es.select(_source=vertex)

    if is_local_std and len(v_out_edges) > 0:
        v_out_edges['weight'] = [tuple(b) for b in scale(v_out_edges['weight']) + 10]

    # Sort based on edge weight for each m features and keep only k members
    kNN_edge_dict = dict()
    for i in range(m):
        kNN_edge_dict[i] = sorted(v_out_edges, key=lambda e:e['weight'][i], reverse = True)[:k]
    #for i, key in enumerate(kNN_edge_dict.keys()):
        #print(key, ':', [(e.index, e['weight'][i]) for e in kNN_edge_dict[key]])

    # Union of all edges
    all_contenders = set.union(*[set(val) for val in kNN_edge_dict.values()])

    # collect all evidences and take average
    kNN_contenders_dict = {e:np.sum([e['weight'][i] for i in range(m) if e in kNN_edge_dict[i]])/m for e in all_contenders}
    kNN_contenders = sorted(kNN_contenders_dict.items(),key=lambda x:x[1],reverse=True)[:k]
    kNN_edges = [t[0] for t in kNN_contenders]

    # Delete remaining edges
    e_to_be_removed = [e.index for e in v_out_edges if e not in kNN_edges]

    # Return kNN vertices
    v_kNN = [e.target for e in kNN_edges]
    #kNN_wt = [e['weight'] for e in kNN_edges]
    #print(vertex, ':', list(zip(v_kNN, kNN_wt)))

    return v_kNN, e_to_be_removed
