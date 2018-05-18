import pandas as pd
import numpy as np
import ast
import sys
import os
import configRead as con
import glob

def postProcessor():
    # Read config files
    print('Reading config parameters for post-processing')
    try:
        mainConfig = con.readMainConfigs ()
        con.setGlobals(mainConfig)
        configParams = con.readConfigFile(con.paramConfigFile)
        configDistributions = con.readConfigFile(con.distributionsFile)
        modeparams = mainConfig[con.mode]
      
    except:
        print('Error in reading config files.. Make sure you have not made any mistake in editing parameters')

    if con.mode != 'M':
        sys.exit('BTS is not operating in mixining mode. Please change the mode in mainConfig.ini file to M.')

    tran_info_path = os.path.join (modeparams['mixed_data'], modeparams['traninfo_folder'])

    # File paths
    vertices = tran_info_path + 'vertices.csv'
    edges = tran_info_path + 'edges.csv'
    metadata_file = modeparams['mixed_data'] + 'combinedMetadata.csv'

    print('Reading combined metadata file..')
    metadata_df = pd.read_csv(metadata_file)
    print('Metadata read for ', len(metadata_df), ' companies.')
    print('Reading vertices and edges..')
    # Read vertices and edges
    #v_set = pd.read_csv(vertices)

    all_files_v = glob.glob(tran_info_path + 'vertices*.csv')
    df_list_v = list()
    for file in all_files_v:
        df = pd.read_csv(file)
        df_list_v.append(df)
    v_set = pd.concat(df_list_v, ignore_index = True)
    all_files_v.clear()
    df_list_v.clear()
    
    all_files = glob.glob(tran_info_path + 'edge*.csv')
    df_list = list()
    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)
    e_set = pd.concat(df_list,  ignore_index = True)
    all_files.clear()
    df_list.clear()

    acc_under_study = list(metadata_df.loc[:,'acc_no'])
    print('Number of edges read = ', len(e_set))
    print('Number of vertices read = ', len(v_set))

    if modeparams['remove_edges'] == 'Y':
        print('Removing edges for outside accounts')
        # Graph with just accounts under study
        
        v_set_under_study = v_set.loc[v_set.vertex.isin(acc_under_study), ['vertex']]
        e_set_under_study = e_set.loc[(e_set.src_acc.isin(acc_under_study)) & (e_set.dest_acc.isin(acc_under_study))]

        # Write reduced vertices and edges
        v_set_under_study.to_csv(modeparams['mixed_data'] + '\\reducedVertices.csv', index=False)
        e_set_under_study.to_csv(modeparams['mixed_data'] + '\\reducedEdges.csv', index=False)
    else:
        print('Collapsing edges for outside accounts')

        # Collapse edges by adding a dummy account
        n = len(v_set)
        #v_set.set_index(np.arange(n))
        v_set_under_study = v_set.loc[v_set.vertex.isin(acc_under_study), ['vertex']]
        #v_set_under_study.set_index(np.arange(len(v_set_under_study)))
        v_set_under_study.loc[len(v_set_under_study), 'vertex'] = 'outside'
        e_set_under_study = e_set.loc[(e_set.src_acc.isin(acc_under_study)) | (e_set.dest_acc.isin(acc_under_study))]
        e_set_under_study.loc[(e_set.src_acc.isin(acc_under_study)) & ~(e_set.dest_acc.isin(acc_under_study)), 'dest_acc'] = 'outside'
        e_set_under_study.loc[(e_set.dest_acc.isin(acc_under_study))& ~(e_set.src_acc.isin(acc_under_study)), 'src_acc'] = 'outside'

        e_set_under_study = e_set_under_study.groupby(by=['src_acc', 'dest_acc'], as_index = False).agg({'amount':np.sum, 'n_tran':np.sum})
        #e_set_under_study.columns = e_set_under_study.columns.droplevel(level=0)
        e_set_under_study.columns = ['src_acc', 'dest_acc', 'amount', 'n_tran']

        # Write reduced vertices and edges
        v_set_under_study.to_csv(modeparams['mixed_data'] + 'collapsedVertices.csv', index=False)
        e_set_under_study.to_csv(modeparams['mixed_data'] + 'collapsedEdges.csv', index=False)

    print('Number of edges reduced to ', len(e_set_under_study))
    print('Number of vertices reduced to ', len(v_set_under_study))

postProcessor()