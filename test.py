import igraph as ig
import pandas as pd
import numpy as np
import ast
import re
from sklearn.preprocessing import scale
from math import log
import pickle
import kNN_all as kNN
import sharedNN as sNN

def load_graph (F, is_local_std):
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
    e_set = e_set.fillna(0.0)
    # Vertices and edges
    v_list = list(v_set['vertex'])
    e_list = [tuple(x) for x in e_set[['src_acc', 'dest_acc']].values]

    
    if not is_local_std:
        e_set[F] = scale(e_set[F], axis=0) + 10
    '''
    for f in F:
        if np.max(e_set[f]) - np.min(e_set[f]) > 10 ** 5:
            e_set[f] = scale(np.log(e_set[f]))
        else:
            e_set[f] = scale(e_set[f])
    '''   
    e_multi = [tuple(x) for x in e_set[F].values]
    G_org = ig.Graph(directed=True)
    G_org.add_vertices(v_list)
    G_org.add_edges(e_list)
    G_org.es['weight'] = e_multi
    G_org.es['label'] = e_multi
    G_org.vs['label'] = v_list

    return G_org

def test (k, m, F, is_local_std):

    G_multi = G_org.copy()

    if m == 1:
        graph_name = r'G:\Study\2017-2018\Work\Experimentation\Results_10_5\SingleEdge' + '\\' + pickle_folder  + '\graph_' + str(k) + '_' + 'NN.pickle'
        kNN_dict_file_name = r'G:\Study\2017-2018\Work\Experimentation\Results_10_5\SingleEdge' + '\\' + pickle_folder + '\\' + str(k) + 'NN_dict.pickle'
    else:
        graph_name = r'G:\Study\2017-2018\Work\Experimentation\Results_10_5\MultiEdge' + '\\' + pickle_folder  + '\graph_' + str(k) + '_' + 'NN.pickle'
        kNN_dict_file_name = r'G:\Study\2017-2018\Work\Experimentation\Results_10_5\MultiEdge' + '\\' + pickle_folder + '\\' + str(k) + 'NN_dict.pickle'

    kNN_dict = dict()
    e_remove = dict()
    
    # Read existing kNN and graph
    with open (kNN_dict_file_name, 'rb') as f:
        kNN_dict = pickle.load(f)

    with open (graph_name, 'rb') as f:
        G_multi = pickle.load(f)
    '''
    if m > 1:
    # Caluculate kNN
        if is_m_way:
            for v in G_multi.vs.indices:
                kNN_dict[v], e_remove[v] = kNN.kNN_m_way(G_multi, v, k,m,is_local_std)
                G_multi.delete_edges(e_remove[v])
        else:
            for v in G_multi.vs.indices:
                kNN_dict[v], e_remove[v] = kNN.kNN_m_avg(G_multi, v, k,m,is_local_std)
                G_multi.delete_edges(e_remove[v])
    else:
        for v in G_multi.vs.indices:
            kNN_dict[v], e_remove[v] = kNN.kNN_single_edge(G_multi, v, k)
            G_multi.delete_edges(e_remove[v])
    
    # Pickle kNN and graph    
    G_multi.save (graph_name)
    with open(kNN_dict_file_name, 'wb') as handle:
        pickle.dump(kNN_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    '''
    if is_ct:
        for kt in range(3, min(k, 10)):
            for ct in ct_list:
                print('k = ', k, 'kt = ',kt,'ct = ', ct)
                cls2, cls3 = sNN.shared_NN_ct(G_multi, kNN_dict, kt, m, ct)
                results.append((k, kt, ct, cls2, cls3))
    else:
        for kt in range(3, min(k, 10)):
            print('k = ', k, 'kt = ',kt)
            cls2, cls3 = sNN.shared_NN(G_multi, kNN_dict, kt, m)
            results.append((k, kt, cls2, cls3))


# Check all input params 
k_list = [4, 5,6,7,8,9,10,11,12]
ct_list = [10,20,30,40,50,60,70,80,90,100]
F = ['amount', 'n_tran']; m = 2

is_local_std = False
is_ct = True
is_m_way = False

pickle_folder = 'amount_n_tran_avg_g'
result_file = 'exp_amount_n_tran_avg_g_cNN.csv'

results = list()

G_org = load_graph(F, is_local_std)
print('Graph loaded. vertices = ', len(G_org.vs), 'edges = ', len(G_org.es))

# run for all k
for k in k_list:
    print('k=',k)
    test (k, m, F, is_local_std)

# Save results

if m > 1:
    exp_res_path = r'G:\Study\2017-2018\Work\Experimentation\Results_10_5\MultiEdge\SharedNN' + '\\' + result_file
else:
    exp_res_path = r'G:\Study\2017-2018\Work\Experimentation\Results_10_5\SingleEdge\SharedNN' + '\\' + result_file

if is_ct:
    exp_res_df = pd.DataFrame(data= results, columns = ['k', 'kt', 'ct', 'sets2', 'sets3'])
    exp_res_df.to_csv(exp_res_path)
else:
    exp_res_df = pd.DataFrame(data= results, columns = ['k', 'kt', 'sets2', 'sets3'])
    exp_res_df.to_csv(exp_res_path)

print('Find results at ', exp_res_path)