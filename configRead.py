#%%
import configparser as cfg
import json
import os
import glob
#%%

mode = 'NA'
paramConfigFile = 'NA'
distributionsFile = 'NA'
metadataFile = 'NA'
tranData = 'NA'
mixedData = 'NA'
tranInfo = 'NA'
additionalFiles = 'NA'

def setGlobals (configParams):
    global mode
    global paramConfigFile
    global distributionsFile
    global metadataFile
    global tranData
    global mixedData
    global tranInfo
    global additionalFiles

    mainconfig = configParams['mainconfig']
    mode = mainconfig['mode']
    modeparams = configParams[mode]

    if mode != 'M':    
        dataPath = modeparams['company_data_path']
        if not os.path.exists(dataPath):
            os.makedirs(dataPath)
        metadataFile = os.path.join(dataPath, modeparams['metadata_file_name'])

        tranDataFolder = modeparams['tran_data']
        tranData = os.path.join(dataPath, tranDataFolder)
        if not os.path.exists(tranData):
            os.makedirs(tranData)

        cwd = os.getcwd()
        paramConfigFile = os.path.join(cwd, mainconfig['param_config_file_name']) 
        distributionsFile = os.path.join(cwd, mainconfig['distributions_file_name']) 

        additionalFiles = os.path.join(dataPath, mainconfig['additional_files_folder'])
        if not os.path.exists(additionalFiles):
            os.makedirs(additionalFiles)
    else:
        mixedDataFolder = modeparams['mixed_data']
        if not os.path.exists(mixedDataFolder):
            os.makedirs(mixedDataFolder)

        honestDataFolder = modeparams['honest_data_folder']
        shellDataFolder = modeparams['shell_data_folder']

        honest_metadata_file = os.path.join(honestDataFolder, modeparams['honest_metadata_file_name'])
        shell_metadata_file = os.path.join(shellDataFolder, modeparams['shell_metadata_file_name'])
        metadataFile = [honest_metadata_file, shell_metadata_file]

        honest_tranInfo = os.path.join(honestDataFolder, modeparams['traninfo_folder'])
        shell_TranInfo = os.path.join(shellDataFolder, modeparams['traninfo_folder'])
        tranInfo = [honest_tranInfo, shell_TranInfo]
        mixedData = modeparams['mixed_data']


def readConfigFile (file_name):
    ''' This function reads config file and returns all sections and options in disctionary form'''

    # Read file 
    con = cfg.ConfigParser()
    con.read (file_name)

    parsedFile = dict()

    sections = con.sections()

    for section in sections:
        options = con.options(section)
        optionDict = dict()
        for option in options:
            optionDict [option] = json.loads(con.get(section, option))

        parsedFile[section] = optionDict

    return parsedFile

def readMainConfigs ():
    cur_working_dir = os.getcwd()
    file_path = os.path.join(cur_working_dir, 'mainConfig.ini')
    configParams = readConfigFile(file_path)
    return configParams

def getParameterDistribution (param_name):
    '''This function exctracts keys and values separately present in a parameter and returns list of keys and values'''

    param_key_list = []
    param_value_list = []

    for key, value in param_name.items():
        param_key_list.append(key)
        param_value_list.append(value)

    return param_key_list, param_value_list

def readAllParams (configFilePath):
    '''This function reads all config files together'''
    allConfigFiles = glob.glob(configFilePath + '\*.ini')
    con = cfg.ConfigParser()
    configurations = dict()

    for f in allConfigFiles:
        con.read (f)
        sections = con.sections()
        for section in sections:
            options = con.options(section)
            for option in options:
                configurations[option] = json.loads(con.get(section, option))

    return configurations



