import numpy as np
import pandas as pd
import configRead as con
import glob as glob
from math import ceil
import ast
import matplotlib.pyplot as plt
import os

# Globals
tran_col_list = ['src_acc', 'src_acc_type', 'dest_acc', 'dest_acc_type', 'currency', 'payment_mode', 'amount', 'date', 'location', 'is_tax_haven', 'tran_label']

def readTransactionData (trandataPath, startId, endId):
    '''This function reads trandata files in chunks'''

    allFiles = [trandataPath + '\\tran{0}.csv' .format(x) for x in range(startId, endId+1)]
    df_list = list()
    tran_data_df = pd.DataFrame(columns=tran_col_list)
    for f in allFiles:
        df = pd.read_csv(f)
        tran_data_df = pd.concat([tran_data_df, df])
        df=df.iloc[0:0]
    return tran_data_df

def prepareSummary (trandata_df):

    # Tran info for graph
    trandata_df['date'] = trandata_df.date.astype(float)
    trandata_df['month_no'] = np.floor(trandata_df['date'] / 30.5)
    trandata_df['month_no'] = trandata_df.month_no.astype(int)
    trandata_df['week_no'] = np.floor(trandata_df['date'] / 7)
    trandata_df['week_no'] = trandata_df.week_no.astype(int)

    tran_info_df = trandata_df.groupby(by=['src_acc', 'dest_acc', 'src_acc_type', 'dest_acc_type', 'date', 'week_no', 'month_no'], as_index = False).agg({'amount':[np.sum, 'count']})
    tran_info_df.columns = tran_info_df.columns.droplevel(level=0)
    tran_info_df.columns = ['src_acc', 'dest_acc','src_acc_type', 'dest_acc_type','date', 'week_no', 'month_no','amount', 'n_tran']

    return tran_info_df

def summarizeData ():

    # Read config files
    print('Reading config parameters for graph summary')
    try:
        mainConfig = con.readMainConfigs ()
        con.setGlobals(mainConfig)
        
        modeparams = mainConfig[con.mode]

    except:
        print('Error in reading config files.. Make sure you have not made any mistake in editing parameters')

    # Read metadata
    metadata_df = pd.DataFrame()
    try:
        metadata_df = pd.read_csv(con.metadataFile, converters={'in_amt_wt':ast.literal_eval, 'supply_amt_wt':ast.literal_eval, 'utility_proportions': ast.literal_eval, 'inside_customers': ast.literal_eval, 'inside_suppliers':ast.literal_eval, 'utility_accs':ast.literal_eval},index_col='index')
    except:
        print('Error in reading metadata')
    
    metadata_df['industry_sector'] = metadata_df.industry_sector.astype('str')
    company_acc = metadata_df['acc_no']
    N = len(metadata_df)
    print('Metadata for ', N, ' companies is read successfully!')

    # Load tran data in chunks prepare summary for small data
    allFiles = glob.glob(con.tranData + '\*.csv')
    n_files = len(allFiles)
  
    if n_files == 0:
        print('No transaction data found!! Make sure trandatapath parameter is set appropriately.')
    else:
        chunk_size = 8
        n_iterations = ceil(n_files / chunk_size)
        # Prepare folder for summarized data
        companyDataPath = modeparams['company_data_path']
        
        tranInfoPath = companyDataPath + 'TranInfo\\'
       
        if not os.path.exists(tranInfoPath):
            os.makedirs(tranInfoPath)
            
        i = 0

        for i in range(n_iterations): 

            # Read chunk
            start_index = i * chunk_size
            end_index = start_index + chunk_size - 1 if i < n_iterations - 1 else n_files -1
            print('Files getting read from ', start_index, end_index)
     
            tran_data_df = readTransactionData(con.tranData, start_index, end_index)

            # Summarize chunk
            partial_traninfo_df = prepareSummary (tran_data_df)

            # Write summary to disk
            file_path = tranInfoPath + 'partialTranInfo' + str(i) + '.csv'
            partial_traninfo_df.to_csv (file_path)

        print('Partial summary files are written..')
        #combinedFiles_df_grouped = combineTranInfoFiles (tranInfoPath)

summarizeData ()