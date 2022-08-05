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

        # output_dim += 1

    #input = TFile.Open('/eos/cms/store/group/phys_higgs/cmshww/amassiro/HWWNano/Summer16_102X_nAODv7_Full2016v7/MCl1loose2016v7__MCCorr2016v7__l2loose__l2tightOR2016v7/nanoLatino_DYJetsToLL_M-50_ext2__part9.root')
    #input = TFile.Open('/eos/user/f/fernanpe/WW_skimmedTrees_nanoAODv7/2016/nanoLatino_DYJetsToLL_M-50_ext2__part9.root')
    #TFile* signal = new TFile("");
    #TTree* sigTree = (TTree*)(signal->Get("tree"));
    #dataloader.AddBackgroundTree(input.Get("Events"), 1.0)

    #TFile* data = new TFile("/eos/cms/store/group/phys_higgs/cmshww/amassiro/HWWNano/Summer16_102X_nAODv7_Full2016v7/MCl1loose2016v7__MCCorr2016v7__l2loose__l2tightOR2016v7/nanoLatino_DYJetsToLL_M-50_ext2__part9.root");
    #TTree* dataTree = (TTree*)(data->Get("Events"));

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
