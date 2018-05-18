import metadatagenerator as m
import trandatagenerator as t
import configRead as config
import os
import time
import dataSummary as ds
import dataSanityCheck as dc
#%%

def main ():

    start_time = time.time()
    print ("Reading input parameters..");
    configParams = config.readMainConfigs()

    #try:
    config.setGlobals(configParams)
    
    N = configParams[config.mode]['n']
    companyIDSeed = configParams[config.mode]['company_id_seed']

    #print('Choose what you wan to do')
    #print('1. Generate metadata\n2. Generate transactions from existing metadata\n3. Perform sanity check')

    m.generateMetadata(N, companyIDSeed)
    print("Metadata generation successful!! Check output file at " + config.metadataFile);

    #except:
    #print ("Error is generating metadata..")
    #try:
    t.generateTransactionData(companyIDSeed)
    end_time = time.time()
    print("Execution time: ", end_time - start_time)
    #except:
    #print ("Error is generating transaction data..")
    start_time = time.time()
    ds.summarizeData()
    end_time = time.time()
    print("Execution time: ", end_time - start_time)

    start_time = time.time()
    #dc.sanityCheck()
    end_time = time.time()
    print("Execution time: ", end_time - start_time)

#%%

main()

#%%
