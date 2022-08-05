from ROOT import TMVA, TFile, TTree, TCut, TChain
import ROOT
import config_cfg as config
 
def filter_events(df):
    """
    Reduce initial dataset to only events which shall be used for training
    """
    return df.Filter("mll>20 && Lepton_pt[0]>25 \
            && Lepton_pt[1]>20 \
            && (fabs(91.1876 - mll) > 15.) \
            && (nLepton>=2 && (Lepton_pt[2],0)<10) \
            && abs(Lepton_eta[0])<2.5 && abs(Lepton_eta[1])<2.5 \
            && ptll>30 \
            && PuppiMET_pt > 20 \
            && ((Lepton_pdgId[0]*Lepton_pdgId[1] == -11*11) || (Lepton_pdgId[0]*Lepton_pdgId[1] == -13*13)) \
            && (CleanJet_pt[0], 0) < 30.",'supercut')

 
def define_variables(df):
    """
    Define the variables which shall be used for training
    """
    return df.Define("Lepton_pt_1", "Lepton_pt[0]")\
             .Define("Lepton_pt_2", "Lepton_pt[1]")\
             .Define("Lepton_eta_1", "Lepton_eta[0]")\
             .Define("Lepton_eta_2", "Lepton_eta[1]")
 
 
Variables = [
    'Lepton_pt_1',
    'Lepton_pt_2',
   'Lepton_eta_1',
   'Lepton_eta_2',
   'TkMET_pt',
   'mpmet',
   'projpfmet',
   'projtkmet',
   'ptll',
   'drll', 
   'dphill',
   'dphilmet',
   'dphillmet',
   'PuppiMET_pt',
    'mth',
   'mtw1',
   'mtw2',
   'pTWW',
   ]
 
if __name__ == "__main__":
    # Load configuration
    with open("configuration.py") as handle:
        exec(handle)

    samples={}
    structure={}
    cuts={}

    for f in [samplesFile, structureFile, cutsFile]:
        with open(f) as handle:
            exec(handle)


    # Reduce sample files for fast dev
    for sampleName, sample in samples.items():
        if sampleName not in ['ggWW_2016','ggWW_2017','ggWW_2018','WW_2016','WW_2017','WW_2018','DY_2016','DY_2017','DY_2018']:
            samples.pop(sampleName)
            continue

        sample['name'] = sample['name']

    cnt_sig = 0
    cnt_back = 0  
    for sampleName, sample in samples.items():
        if config.structure[sampleName]['isData']==1:
            continue
        

        sample['tree'] = TChain("Events")
        for f in sample['name']:
            sample['tree'].Add(f)

        # Load dataset, filter the required events and define the training variables
        df = ROOT.RDataFrame(sample['tree'])
        #df = filter_events(df)
        df = define_variables(df)

        # Book cutflow report
        report = df.Report()
        
        # Split dataset by event number for training and testing
        columns = ROOT.std.vector["string"](Variables)
        if structure[sampleName]['isSignal']==1:
            df.Filter("event % 2 == 0", "Select events with even event number for training")\
            .Snapshot("Events", "XGBOOST/XGtrain_" + str(cnt_sig) +"signal"+ ".root", columns)
            df.Filter("event % 2 == 1", "Select events with even event number for testing")\
            .Snapshot("Events", "XGBOOST/XGtest_" + str(cnt_sig) +"signal"+ ".root", columns)
            cnt_sig+=1
        else:
            df.Filter("event % 2 == 0", "Select events with odd event number for training")\
            .Snapshot("Events", "XGBOOST/XGtrain_" + str(cnt_back) +"background"+ ".root", columns)
            df.Filter("event % 2 == 1", "Select events with odd event number for testing")\
            .Snapshot("Events", "XGBOOST/XGtest_" + str(cnt_back) +"background"+ ".root", columns)
            cnt_back+=1
            
        report.Print()
