import pandas as pd
import ast
import os
import configRead as con
from dataSummary import old_combineTranInfoFiles
from math import ceil

def mixer ():
    # Read config files
    print('Reading config parameters for data summary')
    try:
        mainConfig = con.readMainConfigs ()
        con.setGlobals(mainConfig)
        configParams = con.readConfigFile(con.paramConfigFile)
        configDistributions = con.readConfigFile(con.distributionsFile)
        modeparams = mainConfig[con.mode]
        
    except:
        print('Error in reading config files.. Make sure you have not made any mistake in editing parameters')

    # Read metadata files and join them
    print('Reading both metadata files..')
    metadata_df = pd.DataFrame()
    for file in con.metadataFile:
        df = pd.read_csv(file, converters={'in_amt_wt':ast.literal_eval, 'supply_amt_wt':ast.literal_eval, 'utility_proportions': ast.literal_eval, 'inside_customers': ast.literal_eval, 'inside_suppliers':ast.literal_eval, 'utility_accs':ast.literal_eval},index_col='index')
        metadata_df = pd.concat([metadata_df, df])

    # Write joint metadata file
    combined_metadata_path = con.mixedData + 'combinedMetadata.csv'
    metadata_df.to_csv(combined_metadata_path,index=False)
    print('Metadata is combined successfully! Metadata file can be accessed at ', combined_metadata_path)

    # Combine tranInfo
    print('Combining transaction information files')
    combinedFiles_df_grouped = pd.DataFrame()
    for path in con.tranInfo:
        df = old_combineTranInfoFiles(path)
        combinedFiles_df_grouped = pd.concat([combinedFiles_df_grouped, df])

    # get vertices and edges
    mixed_data_path = os.path.join(modeparams['mixed_data'], modeparams['traninfo_folder'])
    if not os.path.exists(mixed_data_path):
        os.makedirs(mixed_data_path)

    src_acc = set(combinedFiles_df_grouped.loc[:,'src_acc'])
    dest_acc = set(combinedFiles_df_grouped.loc[:,'dest_acc'])
    v = src_acc.union(dest_acc)
    v_df = pd.DataFrame(data = list(v), columns=['vertex'])
    #v_df.to_csv(mixed_data_path + 'vertices.csv', index=False)

    #Write vertices
    file_size_v = 50000
    n_files_v = ceil(len(v_df) / file_size_v)
    i = 0
    while i < n_files_v:
        start_index = i*file_size_v
        end_index = start_index + file_size_v
        vSet = v_df[start_index:end_index]
        vSet.to_csv(mixed_data_path + 'vertices' + str(i) + '.csv', index = False)
        i = i+1

    file_size = 50000
    n_files = ceil(len(combinedFiles_df_grouped) / file_size)
    i = 0
    while i < n_files:
        start_index = i*file_size
        end_index = start_index + file_size
        edgeSet = combinedFiles_df_grouped[start_index:end_index]
        edgeSet.to_csv(mixed_data_path + 'edges' + str(i) + '.csv', index = False)
        i = i+1
    #e = list(combinedFiles_df_grouped.itertuples(index=False,name=None))
    #combinedFiles_df_grouped.to_csv(mixed_data_path + 'edges.csv', index = False)

    print('Transaction data combined successfully! Vertices and edges can be found at ', mixed_data_path)

mixer()