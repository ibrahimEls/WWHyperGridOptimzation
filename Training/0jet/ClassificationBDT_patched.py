#!/usr/bin/env python
from ROOT import TMVA, TFile, TTree, TCut, TChain
from subprocess import call
from os.path import isfile

import config_cfg as config

# Setup TMVA
def runJob():
    TMVA.Tools.Instance()
    TMVA.PyMethodBase.PyInitialize()

    dataloader = TMVA.DataLoader('TMVA_BDT_DY_v1')
    output = TFile.Open('TMVA_BDT_DY.root', 'RECREATE')
    factory = TMVA.Factory('TMVAClassification', output,
            '!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification')
    
    for br in config.mvaVariables:
        dataloader.AddVariable(br)
    
    samplesSig = TChain("Events")
    samplesBkg = TChain("Events")

    for line in open('samples.txt', 'r'):
        if 'WW-LO' in str(line).strip():
            samplesSig.AddFile(str(line).strip())
        else:
            samplesBkg.AddFile(str(line).strip())
    
    dataloader.AddSignalTree(samplesSig, 1.0)
    dataloader.AddBackgroundTree(samplesBkg, 1.0)

    dataloader.PrepareTrainingAndTestTree(TCut(config.cut),'SplitMode=Random::SplitSeed=10:NormMode=EqualNumEvents')

    factory.BookMethod(dataloader, TMVA.Types.kBDT, "BDTG4D3",   "!H:!V:NTrees=500:MinNodeSize=1.5%:BoostType=Grad:Shrinkage=0.05:UseBaggedBoost:GradBaggingFraction=0.5:nCuts=500:MaxDepth=3" );
    factory.BookMethod(dataloader, TMVA.Types.kBDT, "BDTG4C3", "!H:!V:NTrees=500:MinNodeSize=1.5%:BoostType=Grad:Shrinkage=0.05:UseBaggedBoost:GradBaggingFraction=0.5:nCuts=300:MaxDepth=2" );
    factory.BookMethod(dataloader, TMVA.Types.kBDT, "BDTG4SK01",   "!H:!V:NTrees=500:MinNodeSize=1.5%:BoostType=Grad:Shrinkage=0.01:UseBaggedBoost:GradBaggingFraction=0.5:nCuts=500:MaxDepth=2" );
    factory.BookMethod(dataloader, TMVA.Types.kBDT, "BDTG4F07"    ,   "!H:!V:NTrees=500:MinNodeSize=1.5%:BoostType=Grad:Shrinkage=0.05:UseBaggedBoost:GradBaggingFraction=0.7:nCuts=500:MaxDepth=2" );
    factory.BookMethod(dataloader, TMVA.Types.kBDT, "BDTG4SK01F07",   "!H:!V:NTrees=500:MinNodeSize=1.5%:BoostType=Grad:Shrinkage=0.01:UseBaggedBoost:GradBaggingFraction=0.7:nCuts=500:MaxDepth=2" );

    # Run training, test and evaluation
    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()

    output.Close()

if __name__ == "__main__":
    runJob()
