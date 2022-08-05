import subprocess
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from os.path import exists
import json
import shutil
import sys


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
    bst_type = ''
    # updating BDTPARAMS with auc values found in result
    for res in result:
        BDT_name = res[0]
        auc = res[1]
        for boost_type in boost_type_list:
            cnt = 0
            for out in BDTPARAMS[boost_type]:
                if out[0] == BDT_name and BDTPARAMS[boost_type][cnt][2] == 0:
                    length_dict = {key: len(value) for key, value in BDTPARAMS.items()}
                    pos = sum(length_dict.values())

                    BDTPARAMS[boost_type][cnt][2] = auc
                    BDTPARAMS[boost_type][cnt][3] = pos
                    bst_type = boost_type
                cnt +=1

    np.save(BDTPARAMS_file_name,BDTPARAMS)
    return (BDTPARAMS,BDTPARAMS[boost_type][cnt],bst_type)

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
        BDTPARAMS = (np.load(BDTPARAMS_file_name,allow_pickle=True)).item()
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

def merge_files(BDTPARAMS, merge_path,BDTPARAMS_file_name):

    for f in os.listdir(merge_path):
        BDTPARAMS_new = (np.load(merge_path+str(f))).item()
        for key in BDTPARAMS_new.keys():
            if len(BDTPARAMS_new[key])>0:
                for val in BDTPARAMS_new[key]:
                    if val not in BDTPARAMS[key]:
                        BDTPARAMS[key].append(val)
    
    np.save(BDTPARAMS_file_name,BDTPARAMS)
    return BDTPARAMS


def make_plot(BDTPARAMS,split):
    
    all_BDT = []
    for val in BDTPARAMS.values():
        for val2 in val:
            all_BDT.append(val2)
    
    num_bars = int(np.ceil(len(all_BDT)/split))
    cnt = split
    rank = []
    best_auc = []

    for i in range(num_bars):
        rank.append(cnt)
        current_list = [[0,0,0]]

        for j in range(len(all_BDT)):
            if len(all_BDT[j]) >= 4:
                if all_BDT[j][3]<=cnt:
                    current_list.append(all_BDT[j])

        current_list.sort(key = lambda x: x[2],reverse=True)
        best_auc.append(current_list[0][2])

        cnt+=split

    plt.figure()
 
    # creating the bar plot
    plt.plot(rank, best_auc)
    
    plt.xlabel("Number of Architectures Tested")
    plt.ylabel("Current Best AUC")
    plt.title("Number of Architectures Tested vs Current Best AUC")
    plt.show()


if __name__ == "__main__":

    #Params ----------------------------------------------------------------------------
    boost_type_list = ["Grad","RealAdaBoost","Bagging","AdaBoost","RandomForest"] #list of boostypes to iterate through
    
    merge = True
    gen_batch = False # whether or not to genrate more random BDT architectures with condor nodse to evaluate
    gen_more = False # whether or not to genrate more random BDT architectures on current node to evaluate
    gen_best = False # whether or not to generate the best TMVA architecture found so far
    gen_plot = True # whether or not to generate the plot
    try_a_few = False
    

    numgen_pertype = 300  # number of archtectures to genrate per type   
    split = True # whether or not to split genrating TMVA into batches of size split_num
    split_num = 10
    num_per_node = 2

    #insure this matches option in samples_BDT
    useDYHTWWTo2L2Nu = False # what samples to use for training new architectures

    batchQueue = 'tomorrow'
    cwd = os.getcwd()

    #REST FINDINGS THIS WILL DELETE PREVOIUS WORK----------------------------------------------------------------------------
    reset = False
    reset_batch = True
    reset_ranking = False
    reset_temp = True
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
        BDTPARAMS_file_name = "BDTPARAMS_DYHTWWTo2L2Nu"
    else:
        BDTPARAMS_file_name = "BDTPARAMS"
    
    if not gen_batch:
        BDTPARAMS_file_name = BDTPARAMS_file_name+".npy"
        # setting BDTPARAMS dict
        BDTPARAMS = get_params(BDTPARAMS_file_name,reset)

        
    if merge and not gen_batch:
        if useDYHTWWTo2L2Nu:
            BDTPARAMS = merge_files(BDTPARAMS,'./batch_files_DYHT/',BDTPARAMS_file_name)
        else:
            BDTPARAMS = merge_files(BDTPARAMS,'./batch_files/',BDTPARAMS_file_name)

    if gen_batch:
        from LatinoAnalysis.Tools.batchTools  import *
        print("~~~~~~~~~~~ Running mkGrid on Batch Queue")

        if reset_batch:

            if reset_ranking:
                # removing batch npy files
                dir = './batch_files'
                for f in os.listdir(dir):
                    os.remove(os.path.join(dir, f))

            if reset_temp:
                folder = '/eos/user/i/ielshark/temp'
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print('Failed to delete %s. Reason: %s' % (file_path, e))

            # removing job folder
            shutil.rmtree("/afs/cern.ch/user/i/ielshark/cms/HWW/jobs/mkGrid__Random_TMVA_GRID_0jet__ALL",ignore_errors=True)


        if 'slc7' in os.environ['SCRAM_ARCH'] and 'iihe' in os.uname()[1] : use_singularity = True
        else : use_singularity = False

        stepList = ['ALL']
        targetList = []
        batchSplit = ['Target']

        num_times = int(np.ceil(numgen_pertype/num_per_node))
        for boost_type in boost_type_list:
            for i in range(num_times):
                temp_tup = (boost_type,i)
                targetList.append(temp_tup)
    
        tag = 'Random_TMVA_GRID_'
        tag = tag + os.path.basename(cwd)

        jobs = batchJobs('mkGrid',tag,stepList,targetList,','.join(batchSplit),JOB_DIR_SPLIT_READY=True,USE_SINGULARITY=use_singularity)

        jobs.AddPy2Sh()
        jobs.InitPy('import os')
        jobs.InitPy('import subprocess')
        jobs.InitPy("\n")

        for iStep in stepList:
            for iTarget in targetList:
                run_num = np.random.randint(1,10**5)
                if useDYHTWWTo2L2Nu:
                    BDTPARAMS_file_name_new = str(cwd)+'/batch_files_DYHT/'+BDTPARAMS_file_name+str(run_num)+".npy"
                else:
                    BDTPARAMS_file_name_new = str(cwd)+'/batch_files/'+BDTPARAMS_file_name+str(run_num)+".npy"

                instructions_for_configuration_file  = "subprocess.call(['python3','"+str(cwd)+\
                "/mkGrid.py','"+ str(iTarget[0])+  "','"+str(num_per_node)+"','"+str(run_num)+"','"+str(dataloader_name)+"','"+\
                str(BDTPARAMS_file_name_new)+"','"+str(json.dumps((boost_type_list)))+"','"+str(cwd)+"'])\n"

                jobs.AddPy (iStep, iTarget, instructions_for_configuration_file)

        jobs.Sub(batchQueue,'168:00:00',True)
    
     
    # generating more BDT archtectures without batch
    elif gen_more:
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
                result = randomTMVA(boost_type,split_num,run_num,dataloader_name,BDTPARAMS_file_name,str(cwd))
                # updating dictionary with AUC values
                update_dictionary(result,boost_type_list,BDTPARAMS_file_name)

        # getting new BDTPARAMS
        BDTPARAMS = get_params(BDTPARAMS_file_name)

    # genrating best architecture found so far
    elif gen_best:
        
        sorted_params = sort_params(BDTPARAMS,False,BDTPARAMS_file_name,boost_type_list)
        # finding best architecture between all boost types 
        max = 0
        best_str = ""
        list_checked = []
        check = True

        while check:
            for boost_type in boost_type_list:
                if sorted_params[boost_type][0][2]>max and sorted_params[boost_type][0][2]<.97 and sorted_params[boost_type][0] not in list_checked:
                    max = sorted_params[boost_type][0][2]
                    best_str = sorted_params[boost_type][0][1]
                    list_checked.append(sorted_params[boost_type][0])

            print("Creating Best Architecture with AUC value of "+ str(max) + " and params : "+str(best_str))
            # genrating best architecture found so far
            result = randomTMVA("best",best_str,str(0),dataloader_name,BDTPARAMS_file_name,cwd)
            print("Created Best Architecture with re-evualted AUC value of: "+ str(result[0][1])+ " vs its prevously tested AUC of: "+ str(max))
            if np.isclose(result[0][1],max,rtol=0,atol=.01):
                print("Result is correct")
                check = False

    elif gen_plot:
        # sorting BDTPARAMS
        sorted_params = sort_params(BDTPARAMS,False,BDTPARAMS_file_name,boost_type_list)
        make_plot(BDTPARAMS,1)

    elif try_a_few:
        dataloader_name = 'TMVA_BDT_DY_trial' # name of dataloader
        list_params = ["!H:!V:NTrees=993:MaxDepth=9:MinNodeSize=3.72203797391%:nCuts=472:BoostType=Grad:UseBaggedBoost:Shrinkage=0.0112639014049:BaggedSampleFraction=1.43085223552:NodePurityLimit=0.955763453848:SeparationType=CrossEntropy:DoPreselection:SigToBkgFraction=0.57767158687"]

        for params in list_params:
            print("testing: " + params)
            result = randomTMVA("trial",params,str(0),dataloader_name,BDTPARAMS_file_name,cwd)
            BDTPARAMS, result, bst_type = update_dictionary(result,boost_type_list,BDTPARAMS_file_name)
            print("Result for "+ params+ ": "+str(result))
            BDT_small = {bst_type:[result]}
            np.save("batch_files/trial"+str(np.random.randint(1000)),BDT_small)

        sorted_params = sort_params(BDTPARAMS,False,BDTPARAMS_file_name,boost_type_list)

    else:
        sorted_params = sort_params(BDTPARAMS,False,BDTPARAMS_file_name,boost_type_list)

    