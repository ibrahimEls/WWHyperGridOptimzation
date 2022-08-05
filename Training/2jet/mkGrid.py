import subprocess
import numpy as np
import os
from os.path import exists
import sys
import json


def randomTMVA(boostType,numGenerate,run_num,dataloader_name,BDTPARAMS_file_name,cwd):
    """
    Function that calls randomTMVA.py in order to genrate BDT archtectures, train, test and evulate them, then store AUC values in result

    Inputs:
    boostType = boost type to genrate
    numGenerate = number to genrate
    run_num = run_num to put in to BDTstring
    dataloader_name = data loader name
    BDTPARAMS_file_name = file name for numpy file that stores BDTPARAMS

    Output:
    result = list that contains AUC values of genrated architectures
    """

    # running randomTMVA.py
    proc = subprocess.run(['python',cwd+'/randomTMVA.py',boostType,str(numGenerate), str(run_num),dataloader_name,str(BDTPARAMS_file_name),str(cwd)],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc_out = proc.stdout.decode("utf-8")
    proc_error = proc.stderr.decode("utf-8")
    
    for line in proc_error.splitlines():
        sys.stderr.write(line + "\n")
        if 'TFile' in line:
            os.remove(BDTPARAMS_file_name)
            raise Exception(line) 

    # reading STD output and saving AUC values
    cnt = -1
    result = []
    for line in proc_out.splitlines():
        print(line)
        if "-----------" in line and cnt != -1:
            break

        if "ROC-integ" in line:
            cnt += 1
        if cnt>-1:
            if cnt>0:
                auc_ind = [pos for pos, char in enumerate(line) if char == ":"][1] +1
                auc = float(line[auc_ind+1:auc_ind+6])
                
                BDT_name_ind = line.find(dataloader_name) +len(dataloader_name) +1 
                BDT_name = ""
                temp =""
                ind_cnt = 0

                while temp!=" " and temp!=":":
                    BDT_name =  BDT_name + temp
                    temp = line[BDT_name_ind+ind_cnt:BDT_name_ind+ind_cnt+1]
                    ind_cnt +=1
                    
                result.append([BDT_name,auc])
            cnt +=1 

    return result
        
def update_dictionary(result,boost_type_list,BDTPARAMS_file_name):
    """
    Function that updates BDTPARAMS with newely genrated architectures and there AUC value

    Inputs:
    result = list that contains AUC values of genrated archtiectures
    boost_type_list = list of boost types
    BDTPARAMS_file_name = file name for numpy file that stores BDTPARAMS

    Output:
    BDTPARAMS = dictinary that contains AUC values assocaited with genrated archtiectures
    """

    BDTPARAMS = np.load(BDTPARAMS_file_name).item()

    # updating BDTPARAMS with auc values found in result
    for res in result:
        BDT_name = res[0]
        auc = res[1]
        for boost_type in boost_type_list:
            cnt = 0
            for out in BDTPARAMS[boost_type]:
                if out[0] == BDT_name and BDTPARAMS[boost_type][cnt][2] == 0:
                    _, _, files = next(os.walk(os.path.dirname(BDTPARAMS_file_name)))
                    pos = len(files)
                    
                    BDTPARAMS[boost_type][cnt][2] = auc
                    BDTPARAMS[boost_type][cnt][3] = pos
                cnt +=1

    np.save(BDTPARAMS_file_name,BDTPARAMS)
    return BDTPARAMS

def get_params(BDTPARAMS_file_name, reset):
    """
    Function that either genrates or reads BDTPARAMS

    Inputs:
    BDTPARAMS_file_name = file name for numpy file that stores BDTPARAMS

    Output:
    BDTPARAMS = dictinary that contains AUC values assocaited with genrated archtiectures
    """
    
    boost_type_list = ["Grad","RealAdaBoost","Bagging","AdaBoost","RandomForest","XGBoost"]

    # if BDTPARAMS_file_name exists return it
    if exists(BDTPARAMS_file_name) and not reset:
        BDTPARAMS = (np.load(BDTPARAMS_file_name)).item()
    # if not create BDTPARAMS and save it to BDTPARAMS_file_name
    else:
        BDTPARAMS = {}
        for i in boost_type_list:
            BDTPARAMS[i] = []
        np.save(BDTPARAMS_file_name,BDTPARAMS)

    return BDTPARAMS

def sort_params(BDTPARAMS,printone,BDTPARAMS_file_name,boost_type_list=[], boost_type1 = ""):
    """
    Function that sorts params and prints best architechtures for each boost type

    Inputs:
    BDTPARAMS = dictinary that contains AUC values assocaited with genrated archtiectures
    printone = Boolean the decied wheter or not to print one boosttype or more
    BDTPARAMS_file_name = file name for numpy file that stores BDTPARAMS
    boost_type_list = list of boost types
    boost_type1 = boost type to print if printone is true

    Output:
    BDTPARAMS = sorted dictinary that contains AUC values assocaited with genrated archtiectures
    """

    # only printing result for one Boost Type
    if printone:
        current_list = list(BDTPARAMS[boost_type1])

        current_list.sort(key = lambda x: x[2],reverse=True)
        BDTPARAMS[boost_type] = current_list

        best = current_list[0]
        print("Out of "+str(len(current_list))+ " models tested, the Best AUC value for BDT with boost type "+ boost_type+" is "+str(best[2])+ " with parameters of " + str(best[1]))
        print(" ")
    
    # sorting and printing architechtures for all boost types
    else:
        for boost_type in boost_type_list:
            current_list = list(BDTPARAMS[boost_type])

            cnt = 0
            for res in current_list:
                if type(res[2]) == type([]):
                    current_list.pop(cnt)
                elif (res[2]) == 0:
                    current_list.pop(cnt)
                cnt+=1

            current_list.sort(key = lambda x: x[2],reverse=True)
            BDTPARAMS[boost_type] = current_list

            if len(current_list)>0:
                best = current_list[0]
                print("Out of "+str(len(current_list))+ " models tested, the Best AUC value for BDT with boost type "+ boost_type+" is "+str(best[2])+ " with parameters of " + str(best[1]))
                print(" ")

        np.save(BDTPARAMS_file_name,BDTPARAMS)
    return BDTPARAMS

if __name__ == "__main__":
    os.chdir(str(sys.argv[7]))
    result = randomTMVA(str(sys.argv[1]),str(sys.argv[2]),int(sys.argv[3]),str(sys.argv[4]),str(sys.argv[5]),str(sys.argv[7]))
    update_dictionary(result,list(json.loads(sys.argv[6])),str(sys.argv[5]))
    os.remove('/eos/user/i/ielshark/temp/TMVA_BDT_DY_'+str(sys.argv[3])+'.root')
