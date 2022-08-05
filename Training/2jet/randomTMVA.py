#!/usr/bin/env python
from ROOT import TMVA, TFile, TTree, TCut, TChain
from subprocess import call
from os.path import isfile
import numpy as np
import sys
import ROOT
import pickle
import os
from os.path import exists

import config_cfg as config
#from XGBoost_training import XGBoost_Train, load_all_data, Variables

# Setup TMVA
def optimize(boost_type, numGrid,run_num,dataloader_name,BDTPARAMS_file_name,cwd):
    """
    Function that genrates BDT archtectures then trains, tests and evulates them, then store AUC values in result

    Inputs:
    boost_type = boost type to genrate
    numGrid = number to genrate (if boost_type = best numGrid will equal the booking string)
    run_num = run_num to put in to BDTstring
    dataloader_name = data loader name
    BDTPARAMS_file_name = file name for numpy file that stores BDTPARAMS

    Output: N/A
    """

    if boost_type != "best":
        boost_type_list = ["Grad","RealAdaBoost","Bagging","AdaBoost","RandomForest"] #list of boostypes to iterate through
        # configuring TMVA and BDTPARAMS
        if exists(BDTPARAMS_file_name):
            BDTPARAMS = (np.load(BDTPARAMS_file_name)).item()
        # if not create BDTPARAMS and save it to BDTPARAMS_file_name
        else:
            BDTPARAMS = {}
            for i in boost_type_list:
                BDTPARAMS[i] = []

    TMVA.Tools.Instance()
    TMVA.PyMethodBase.PyInitialize()
    
    # configuring dataloader
    dataloader = TMVA.DataLoader(dataloader_name)
    if boost_type != "best": 
        output = TFile.Open('/eos/user/i/ielshark/temp/TMVA_BDT_DY_'+str(run_num)+'.root', 'RECREATE')
    else:
        output = TFile.Open('TMVA_BDT_DY_.root', 'RECREATE')
    factory = TMVA.Factory('TMVAClassification', output,
            '!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification')

    for br in config.mvaVariables:
        dataloader.AddVariable(br)

    # configuring training samples
    for sampleName, sample in config.samples.items():
        if config.structure[sampleName]['isData']==1:
            continue
            
        sample['tree'] = TChain("Events")
        for f in sample['name']:
            sample['tree'].Add(f)

        print('Samples = ')
        print(sample['tree'])
        print(sample.keys())
        if config.structure[sampleName]['isSignal']==1:
            dataloader.AddSignalTree(sample['tree'], 1.0)
            dataloader.SetSignalWeightExpression(sample['weight'])
        else:
            dataloader.AddBackgroundTree(sample['tree'], 1.0)
            dataloader.SetBackgroundWeightExpression(sample['weight'])

    dataloader.PrepareTrainingAndTestTree(TCut(config.cut),'SplitMode=Random::SplitSeed=10:NormMode=EqualNumEvents')

    # param options for diffrent boost types
    SeparationTypedict = {1:"GiniIndex",2:"CrossEntropy",3:"GiniIndexWithLaplace",4:"MisClassificationError",5:"SDivSqrtSPlusB",6:"RegressionVariance"}
    true_false_dict = {0:"False",1:""}

    # if genterating more
    if boost_type != "best":

        # Book numGrid architechtures
        for i in range(numGrid):

            nameStruct = []
            # generating pars and name struct depedning on boosttype
            if boost_type == "Grad":
                nameStruct = ["!H:!V:NTrees=",":MaxDepth=",":MinNodeSize=",":nCuts=",":BoostType=",":UseBaggedBoost",":Shrinkage=",
                ":BaggedSampleFraction=",":NodePurityLimit=",":SeparationType=",":UseFisherCuts",":DoPreselection",":RenormByClass",":SigToBkgFraction="]

                pars = [0 for i in range(14)] 
                pars[0] = np.random.randint(500,1000)
                pars[1] = np.random.randint(3,10)
                pars[2] = str(7*np.random.rand())+"%"
                pars[3] = np.random.randint(10,700)
                pars[4] = boost_type
                pars[5] = true_false_dict[np.random.randint(0,len(true_false_dict.keys()))]
                pars[6] = 2*np.random.rand()
                if pars[5] == "False":
                    pars[7] = pars[5]
                else:
                    pars[7] = 2*np.random.rand()
                pars[8] = np.random.rand()
                pars[9] = SeparationTypedict[np.random.randint(1,len(SeparationTypedict.keys()))]
                pars[10] = "False"
                pars[11] = true_false_dict[np.random.randint(0,len(true_false_dict.keys()))]
                pars[12] = "False"
                pars[13] = 2*np.random.rand()+.5
        
            elif boost_type == "RealAdaBoost":
                nameStruct = ["!H:!V:NTrees=",":MaxDepth=",":MinNodeSize=",":nCuts=",":BoostType=",":UseBaggedBoost",":AdaBoostBeta=",
                ":UseYesNoLeaf",":NodePurityLimit=",":SeparationType=",":UseFisherCuts",":DoPreselection",":RenormByClass"]

                pars = [0 for i in range(13)] 
                pars[0] = np.random.randint(500,1000)
                pars[1] = np.random.randint(3,10)
                pars[2] = str(7*np.random.rand())+"%"
                pars[3] = np.random.randint(10,700)
                pars[4] = boost_type
                pars[5] = "False"
                pars[6] = np.random.rand()
                pars[7] = true_false_dict[np.random.randint(0,len(true_false_dict.keys()))]
                pars[8] = np.random.rand()
                pars[9] = SeparationTypedict[np.random.randint(1,len(SeparationTypedict.keys()))]
                pars[10] = "False"
                pars[11] = true_false_dict[np.random.randint(0,len(true_false_dict.keys()))]
                pars[12] = "False"

            elif boost_type == "Bagging":
                nameStruct = ["!H:!V:NTrees=",":MaxDepth=",":MinNodeSize=",":nCuts=",":BoostType=",":UseBaggedBoost",":Shrinkage=",
                ":BaggedSampleFraction=",":NodePurityLimit=",":SeparationType=",":UseFisherCuts",":DoPreselection",":RenormByClass",":SigToBkgFraction="]

                pars = [0 for i in range(14)] 
                pars[0] = np.random.randint(500,1000)
                pars[1] = np.random.randint(3,10)
                pars[2] = str(7*np.random.rand())+"%"
                pars[3] = np.random.randint(10,700)
                pars[4] = boost_type
                pars[5] = "False"
                pars[6] = "False"
                pars[7] = 2*np.random.rand()
                pars[8] = np.random.rand()
                pars[9] = SeparationTypedict[np.random.randint(1,len(SeparationTypedict.keys()))]
                pars[10] = "False"
                pars[11] = true_false_dict[np.random.randint(0,len(true_false_dict.keys()))]
                pars[12] = "False"
                pars[13] = 2*np.random.rand()+.5

            elif boost_type == "AdaBoost":
                nameStruct = ["!H:!V:NTrees=",":MaxDepth=",":MinNodeSize=",":nCuts=",":BoostType=",":UseBaggedBoost",":AdaBoostBeta=",
                ":UseYesNoLeaf",":NodePurityLimit=",":SeparationType=",":UseFisherCuts",":DoPreselection",":RenormByClass",":SigToBkgFraction="]

                pars = [0 for i in range(14)] 
                pars[0] = np.random.randint(500,1000)
                pars[1] = np.random.randint(3,10)
                pars[2] = str(7*np.random.rand())+"%"
                pars[3] = np.random.randint(10,700)
                pars[4] = boost_type
                pars[5] = "False"
                pars[6] = np.random.rand()
                pars[7] = "False"
                pars[8] = np.random.rand()
                pars[9] = SeparationTypedict[np.random.randint(1,len(SeparationTypedict.keys()))]
                pars[10] = "False"
                pars[11] = true_false_dict[np.random.randint(0,len(true_false_dict.keys()))]
                pars[12] = "False"
                pars[13] = 2*np.random.rand()+.5

            elif boost_type == "RandomForest":
                nameStruct = ["!H:!V:NTrees=",":MaxDepth=",":MinNodeSize=",":nCuts=",":BoostType=",":UseRandomisedTrees",":UseNvars=",
                ":BaggedSampleFraction=",":NodePurityLimit=",":SeparationType=",":UseFisherCuts",":DoPreselection",":RenormByClass",":SigToBkgFraction=",":UsePoissonNvars"]

                pars = [0 for i in range(15)] 
                pars[0] = np.random.randint(500,1000)
                pars[1] = np.random.randint(3,10)
                pars[2] = str(7*np.random.rand())+"%"
                pars[3] = np.random.randint(10,700)
                pars[4] = "Bagging"
                pars[5] = ""
                pars[6] = np.random.randint(1,6)
                pars[7] = 2*np.random.rand()
                pars[8] = np.random.rand()
                pars[9] = SeparationTypedict[np.random.randint(1,len(SeparationTypedict.keys()))]
                pars[10] = "False"
                pars[11] = true_false_dict[np.random.randint(0,len(true_false_dict.keys()))]
                pars[12] = "False"
                pars[13] = 2*np.random.rand()+.5
                pars[14] = true_false_dict[np.random.randint(0,len(true_false_dict.keys()))]
                
            elif boost_type == "XGBoost":
                # pars = [0 for i in range(2)] 
                # pars[0] = np.random.randint(3,10)
                # pars[1] = np.random.randint(500,1000)

                # BDTstring = genertateBDTstring(boost_type,nameStruct,pars,i,run_num)
                # XGBoost_Train(pars,BDTstring[0])
                # x, y_true, w = load_all_data(False)
    
                # # Load trained model
                # File = "XGBOOST/"+name+".root"
                # if (ROOT.gSystem.AccessPathName(File)) :
                #     ROOT.Info("tmva102_Testing.py", File+"does not exist")
                #     exit()
                
                # bdt = ROOT.TMVA.Experimental.RBDT[""](name, File)
    
                # # Make prediction
                # y_pred = bdt.Compute(x)
                
                # # Compute ROC using sklearn
                # from sklearn.metrics import roc_curve, auc
                # fpr, tpr, _ = roc_curve(y_true, y_pred, sample_weight=w)
                # score = auc(fpr, tpr, reorder=True)

                # BDTPARAMS[boost_type].append([BDTstring[0],BDTstring[1],score])
                nameStruct = ["max_depth=",":n_estimators="]
            
            # genterating BDT String
            BDTstring = genertateBDTstring(boost_type,nameStruct,pars,np.random.randint(1,10**5),run_num)
            BDTPARAMS[boost_type].append([BDTstring[0],BDTstring[1],0,0])

            # printing progress
            print("Training " +BDTstring[0]+" with parameters "+BDTstring[1])
            factory.BookMethod(dataloader, TMVA.Types.kBDT, BDTstring[0], BDTstring[1])
           
    # if best then book only the best architechure
    else:
        factory.BookMethod(dataloader, TMVA.Types.kBDT, "BEST_RESULT", numGrid)

    # Run training, test and evaluation
    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()
    np.save(BDTPARAMS_file_name,BDTPARAMS)

    output.Close()

def genertateBDTstring(boostype,nameStruct,pars,num,run_num):
    """
    Function that genrates BDT string for diffrent archtectures

    Inputs:
    boostype = boost type for string
    nameStruct = list that defines struct for name
    pars = list of parameteres
    num = num to use in BDTstring
    run_num = run_num to put in to BDTstring

    Output: N/A
    """

    # generating architechture name
    BDTstring = ["" for i in range(2)] 
    BDTstring[0] = "BDT_"+boostype+"_test_run"+str(run_num)+"_test_"+str(num)

    # generating booking string
    for i in range(len(nameStruct)):
        if pars[i] != "False":
            BDTstring[1] = BDTstring[1]+nameStruct[i]+str(pars[i])

    return BDTstring


if __name__ == "__main__":
    # checking whether to genrate many TMVA or just one best one
    if str(sys.argv[1]) == "best":
        optimize(str(sys.argv[1]),str(sys.argv[2]),int(sys.argv[3]),str(sys.argv[4]),str(sys.argv[5]),str(sys.argv[6]))
    else:
        os.chdir("/eos/user/i/ielshark/temp/")
        optimize(str(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),str(sys.argv[4]),str(sys.argv[5]),str(sys.argv[6]))
