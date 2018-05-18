#%%
import igraph as ig
import pandas as pd
import numpy as np
import ast
import re
from sklearn.preprocessing import scale
from math import log
import pickle
import kNN_all as kNN
#%%
def cluster_of (clusters, vertex):
    for cl in clusters:
        if vertex in cl:
            return cl

def shared_NN (G, kNN_dict, kt, m):
    ''' This function implements shared_NN algorithm. Input k is to find kNN neighbors, kt is to check common neighbors'''
    # Standardize

    # Sparcify
    #kNN_dict = dict()
    #e_remove = dict()
    #for v in G.vs.indices:
        #kNN_dict[v], e_remove[v] = kNN(G, v, k, m)
        #G.delete_edges(e_remove[v])

    # Each vertex is a singleton cluster
    clusters = [{v} for v in G.vs.indices]
    #print(clusters)
    for u in G.vs.indices:
        for v in G.vs.indices:
            if u in kNN_dict[v] and v in kNN_dict[u]:
                #print('pair found ', u,',',v)
                if not set({v}).issubset(cluster_of(clusters, u)):
                    if len(set.intersection(set(kNN_dict[u]), set(kNN_dict[v]))) >= kt: 
                        c_u = cluster_of(clusters, u)
                        c_v = cluster_of(clusters, v)
                        clusters.remove(c_u)
                        clusters.remove(c_v)
                        clusters.append(c_u.union(c_v))
                        print('merged', c_u, c_v)
                        
    clusters_siz_filter_2 = [c for c in clusters if len(c) > 2]
    clusters_siz_filter_3 = [c for c in clusters if len(c) > 3]

    return clusters_siz_filter_2, clusters_siz_filter_3

def kt_compatibility (u, v, kt,kNN_dict):
    return True if len(set.intersection(set(kNN_dict[u]), set(kNN_dict[v]))) >= kt else False


def kt_ct_compatibility (u, cluster_v, kt, ct, kNN_dict):
    return True if len([v for v in cluster_v if kt_compatibility(u, v, kNN_dict)]) / len(cluster_v) * 100 >= ct else False

def kt_ct_pt_compatibility (cluster_u, cluster_v, kt, ct, pt, kNN_dict):
    return True if len([(u,v) for u in cluster_u if kt_ct_compatibility(u, cluster_v, kt, ct, kNN_dict)]) / len(cluster_u) * 100 >= pt else False

def cohesion_compatible_1 (cluster_u, cluster_v, kt, ct, kNN_dict):
    compatibility = [(u, v) for u in cluster_u for v in cluster_v if kt_compatibility(u,v,kt,kNN_dict)] 
    return True if len(compatibility) / (len(cluster_u)*len(cluster_v)) * 100 >= ct else False 

def cohesion_compatible_2 (cluster_u, cluster_v, kt, pt, ct, kNN_dict):
    compatibility_u_v = kt_ct_pt_compatibility (cluster_u, cluster_v, kt, ct, pt, kNN_dict)
    compatibility_v_u = kt_ct_pt_compatibility (cluster_v, cluster_u, kt, ct, pt, kNN_dict)

    return True if compatibility_u_v and compatibility_v_u else False 


def shared_NN_ct (G, kNN_dict, kt, m, ct):
    ''' This function implements shared_NN algorithm. Input k is to find kNN neighbors, kt is to check common neighbors'''
    # Standardize

    # Sparcify
    #kNN_dict = dict()
    #e_remove = dict()
    #for v in G.vs.indices:
        #kNN_dict[v], e_remove[v] = kNN(G, v, k, m)
        #G.delete_edges(e_remove[v])

    # Each vertex is a singleton cluster
    clusters = [{v} for v in G.vs.indices]
    #print(clusters)
    for u in G.vs.indices:
        for v in G.vs.indices:
            if u in kNN_dict[v] and v in kNN_dict[u]:
                c_u = cluster_of(clusters, u)
                if not set({v}).issubset(c_u):
                    c_v = cluster_of(clusters, v)
                    if cohesion_compatible_1 (c_v, c_u, kt, ct, kNN_dict):
              
                        #print('Compatible:', c_u,c_v)
                        clusters.remove(c_u)
                        clusters.remove(c_v)
                        clusters.append(c_u.union(c_v))
                        print('merged', c_u, c_v)
                        
    clusters_siz_filter_2 = [c for c in clusters if len(c) > 2]
    clusters_siz_filter_3 = [c for c in clusters if len(c) > 3]

    return clusters_siz_filter_2, clusters_siz_filter_3


def test (k, m, F, is_local_std):

    G_multi = G_org.copy()
    graph_name = r'G:\Study\2017-2018\Work\Experimentation\Results_10_5\SingleEdge\mean_amount_re' + '\graph_' + str(k) + '_' + 'NN.pickle'
    kNN_dict_file_name = r'G:\Study\2017-2018\Work\Experimentation\Results_10_5\SingleEdge\mean_amount_re' + '\\' + str(k) + 'NN_dict.pickle'

    kNN_dict = dict()
    e_remove = dict()
    for v in G_multi.vs.indices:
        kNN_dict[v], e_remove[v] = kNN.kNN_single_edge(G_multi, v, k)
        G_multi.delete_edges(e_remove[v])
        
    G_multi.save (graph_name)
  
    with open(kNN_dict_file_name, 'wb') as handle:
        pickle.dump(kNN_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for kt in range(3, min(k, 10)):
        print('k = ', k, 'kt = ',kt)
        cls2, cls3 = shared_NN(G_multi, kNN_dict, kt,m)
        results.append((k,kt,cls2, cls3))

'''
parent_path = r'G:\Study\2017-2018\Work\Data'
metadatafile_name = r'\combinedMetadata.csv'
main_folder = r'\Mixed_10000_200_0.05_0.2_10_5'

metadata_file = parent_path + main_folder + metadatafile_name
metadata_df = pd.read_csv(metadata_file, converters={'industry_sector':ast.literal_eval})
metadata_df['industry_sector'] = metadata_df.industry_sector.astype('str')

# Read edges and vertices file

e_set = pd.read_csv(parent_path + main_folder + '\\reducedEdges.csv')
v_set = pd.read_csv(parent_path + main_folder + '\\reducedVertices.csv')
e_set['mean_amount'] = e_set['amount'] / e_set['n_tran']
# Vertices and edges
v_list = list(v_set['vertex'])
e_list = [tuple(x) for x in e_set[['src_acc', 'dest_acc']].values]

k_list = [4, 5,6,7,8,9,10,11,12]
F = ['amount']; m = 1
is_local_std = False

if not is_local_std:
    #e_set[F] = scale(e_set[F], axis=0) + 10
    
    for f in F:
        if np.max(e_set[f]) - np.min(e_set[f]) > 10 ** 5:
            e_set[f] = scale(np.log(e_set[f]))
        else:
            e_set[f] = scale(e_set[f])
    
e_multi = [tuple(x) for x in e_set[F].values]
G_org = ig.Graph(directed=True)
G_org.add_vertices(v_list)
G_org.add_edges(e_list)
G_org.es['weight'] = e_multi
G_org.es['label'] = e_multi
G_org.vs['label'] = v_list

for k in k_list:
    print('k=',k)
    test (k, m, F, is_local_std)

exp_res_path = r'G:\Study\2017-2018\Work\Experimentation\Results_10_5\SingleEdge\SharedNN\exp_mean_amount_re.csv'
exp_res_df = pd.DataFrame(data= results, columns = ['k', 'kt', 'sets'])
exp_res_df.to_csv(exp_res_path)
'''