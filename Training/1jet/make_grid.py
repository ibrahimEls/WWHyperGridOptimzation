import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from os.path import exists


def randomTMVA(boostType,numGenerate,run_num,dataloader_name,BDTPARAMS_file_name):
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
    proc = subprocess.run(['python','randomTMVA.py',boostType,str(numGenerate), str(run_num),dataloader_name,str(BDTPARAMS_file_name)],stdout=subprocess.PIPE)
    proc_out = proc.stdout.decode("utf-8")

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
                    BDTPARAMS[boost_type][cnt][2] = auc
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

    #Params ----------------------------------------------------------------------------
    boost_type_list = ["Grad","RealAdaBoost","Bagging","AdaBoost","RandomForest"] #list of boostypes to iterate through
    gen_more = False # whether or not to genrate more random BDT architectures to evaluate
    split = True # whether or not to split genrating TMVA into batches of size split_num
    split_num = 10
    numgen_pertype = 50  # number of archtectures to genrate per type    
    #insure this matches option in samples_BDT
    useDYHTWWTo2L2Nu = False # what samples to use for training new architectures

    gen_best = True # whether or not to generate the best TMVA architecture found so far

    #REST FINDINGS THIS WILL DELETE PREVOIUS WORK----------------------------------------------------------------------------
    reset = False
    #Rest ----------------------------------------------------------------------------

    if gen_best:
        if useDYHTWWTo2L2Nu:
            dataloader_name = 'TMVA_BDT_DYHT_best' # name of dataloader
        else:
            dataloader_name = 'TMVA_BDT_DY_best' # name of dataloader
    else:
        if useDYHTWWTo2L2Nu:
             dataloader_name = 'TMVA_BDT_DYHT_training' # name of dataloader
        else:
            dataloader_name = 'TMVA_BDT_DY_training' # name of dataloader


    # Main Logic ----------------------------------------------------------------------------
    # setting BDTPARAMS File name
    if useDYHTWWTo2L2Nu:
        BDTPARAMS_file_name = "BDTPARAMS_DYHTWWTo2L2Nu.npy"
    else:
        BDTPARAMS_file_name = "BDTPARAMS.npy"

    # setting BDTPARAMS dict
    BDTPARAMS = get_params(BDTPARAMS_file_name,reset)

    # generating more BDT archtectures
    if gen_more:
        # spliting into batches
        if split:
            num_times = int(np.ceil(numgen_pertype/split_num))
        else:
            num_times = 1
            split_num = numgen_pertype

        # generating numgen_pertype BDT architectures for each Boost type
        for boost_type in boost_type_list:
            for num in range(num_times):
                run_num = np.random.randint(1,10**5)
                # genrating split_num number of BDT architectures
                result = randomTMVA(boost_type,split_num,run_num,dataloader_name,BDTPARAMS_file_name)
                # updating dictionary with AUC values
                update_dictionary(result,boost_type_list,BDTPARAMS_file_name)

        # getting new BDTPARAMS
        BDTPARAMS = get_params(BDTPARAMS_file_name)
    
    # sorting BDTPARAMS
    sorted_params = sort_params(BDTPARAMS,False,BDTPARAMS_file_name,boost_type_list)

    # genrating best architecture found so far
    if gen_best:

        # finding best architecture between all boost types 
        max = 0
        best_str = ""
        for boost_type in boost_type_list:
            if sorted_params[boost_type][0][2]>max:
                max = sorted_params[boost_type][0][2]
                best_str = sorted_params[boost_type][0][1]

        # genrating best architecture found so far
        subprocess.run(['python','randomTMVA.py',"best",str(best_str), str(0),dataloader_name,str(BDTPARAMS_file_name)],stdout=subprocess.PIPE)