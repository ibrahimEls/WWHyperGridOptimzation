import os
import inspect

configurations = os.path.realpath(inspect.getfile(inspect.currentframe())) # this file
configurations = os.path.dirname(configurations) # inclusive BDT Training 0j merged
configurations = os.path.dirname(configurations) # Merged Training 
configurations = os.path.dirname(configurations) # FullRunII
configurations = os.path.dirname(configurations) # WW
configurations = os.path.dirname(configurations) # Configurations


from LatinoAnalysis.Tools.commonTools import getSampleFiles, getBaseW, getBaseWnAOD, addSampleWeight
def nanoGetSampleFiles(inputDir, sample):
    try:
        if _samples_noload:
            return []
    except NameError:
        pass

    return getSampleFiles(inputDir, sample, True, 'nanoLatino_')

#1;95;0c samples
try:
    len(samples)
except NameError:
    import collections
    samples = collections.OrderedDict()


################################################         
################# SKIMS ########################
################################################

mcProduction_2016 = 'Summer16_102X_nAODv7_Full2016v7'
mcSteps_2016 = 'MCl1loose2016v7__MCCorr2016v7__l2loose__l2tightOR2016v7__Skim2016{var}'

mcProduction_2017 = 'Fall2017_102X_nAODv7_Full2017v7'
mcSteps_2017 = 'MCl1loose2017v7__MCCorr2017v7__l2loose__l2tightOR2017v7__Skim2017{var}'

mcProduction_2018 = 'Autumn18_102X_nAODv7_Full2018v7'
mcSteps_2018 = 'MCl1loose2018v7__MCCorr2018v7__l2loose__l2tightOR2018v7__Skim2018{var}'

##############################################
###### Tree base directory for the site ######
##############################################

SITE=os.uname()[1]

if    'iihe' in SITE:
  treeBaseDir = '/pnfs/iihe/cms/store/user/xjanssen/HWW2015'
elif  'cern' in SITE:
  treeBaseDir = '/eos/cms/store/group/phys_muon/fernanpe/TnPTrees'

def makeMCDirectory_2016(var=''):
    if var:
        return os.path.join(treeBaseDir, mcProduction_2016, mcSteps_2016.format(var='__' + var))
    else:
        return os.path.join(treeBaseDir, mcProduction_2016, mcSteps_2016.format(var=''))
		
def makeMCDirectory_2017(var=''):
    if var:
        return os.path.join(treeBaseDir, mcProduction_2017, mcSteps_2017.format(var='__' + var))
    else:
        return os.path.join(treeBaseDir, mcProduction_2017, mcSteps_2017.format(var=''))
		
def makeMCDirectory_2018(var=''):
    if var:
        return os.path.join(treeBaseDir, mcProduction_2018, mcSteps_2018.format(var='__' + var))
    else:
        return os.path.join(treeBaseDir, mcProduction_2018, mcSteps_2018.format(var=''))

mcDirectory_2016 = makeMCDirectory_2016()
mcDirectory_2017 = makeMCDirectory_2017()
mcDirectory_2018 = makeMCDirectory_2018()


################################################
############ BASIC MC WEIGHTS ##################
################################################

Jet_PUIDSF = 'TMath::Exp(Sum$((Jet_jetId>=2)*TMath::Log(Jet_PUIDSF_loose)))'

SFweight = ' * '.join(['SFweight2l','Jet_PUIDSF']) 

PromptGenLepMatch2l = 'Alt$(Lepton_promptgenmatched[0]*Lepton_promptgenmatched[1], 0)'

mcCommonWeight = 'XSWeight*' + PromptGenLepMatch2l + '*METFilter_MC'


###########################################
#############  BACKGROUNDS  ###############
###########################################

useDYHTWWTo2L2Nu = True

if not useDYHTWWTo2L2Nu:
    #xxxxxxx DY xxxxxxx
    #xxxxxxx 2016 xxxxxxx

    files = nanoGetSampleFiles(mcDirectory_2016, 'DYJetsToLL_M-50_ext2') + \
            nanoGetSampleFiles(mcDirectory_2016, 'DYJetsToLL_M-10to50')


    samples['DY_2016'] = {
        'name': files,
        'weight': mcCommonWeight + '*( !(Sum$(PhotonGen_isPrompt==1 && PhotonGen_pt>15 && abs(PhotonGen_eta)<2.6) > 0 &&\
                                    Sum$(LeptonGen_isPrompt==1 && LeptonGen_pt>15)>=2) )',
        'FilesPerJob': 4,
        'suppressNegative' :['all'],
        'suppressNegativeNuisances' :['all'],
        }

    #xxxxxxx 2017 xxxxxxx

    files = nanoGetSampleFiles(mcDirectory_2017, 'DYJetsToLL_M-50_ext1') + \
            nanoGetSampleFiles(mcDirectory_2017, 'DYJetsToLL_M-10to50-LO_ext1')

    samples['DY_2017'] = {
        'name': files,
        'weight': mcCommonWeight + "*( !(Sum$(PhotonGen_isPrompt==1 && PhotonGen_pt>15 && abs(PhotonGen_eta)<2.6) > 0 &&\
                                        Sum$(LeptonGen_isPrompt==1 && LeptonGen_pt>15)>=2) )",
        'FilesPerJob': 8,
        'suppressNegative' :['all'],
        'suppressNegativeNuisances' :['all'],
        }

    #xxxxxxx 2018 xxxxxxx

    files = nanoGetSampleFiles(mcDirectory_2018, 'DYJetsToLL_M-10to50-LO_ext1') + \
            nanoGetSampleFiles(mcDirectory_2018, 'DYJetsToLL_M-50_ext2')

    samples['DY_2018'] = {
        'name': files,
        'weight': mcCommonWeight + '*( !(Sum$(PhotonGen_isPrompt==1 && PhotonGen_pt>15 && abs(PhotonGen_eta)<2.6) > 0 &&\
                                    Sum$(LeptonGen_isPrompt==1 && LeptonGen_pt>15)>=2) )',
        'FilesPerJob': 6,
        'suppressNegative' :['all'],
        'suppressNegativeNuisances' :['all'],
    }

#xxxxxxxxx DY HT xxxxxxxxx
else:
    #xxxxxxx 2016 xxxxxxx
    files = nanoGetSampleFiles(mcDirectory_2016, 'DYJetsToLL_M-50_ext2') \
                            + nanoGetSampleFiles(mcDirectory_2016, 'DYJetsToLL_M-10to50') \
                            + nanoGetSampleFiles(mcDirectory_2016, 'DYJetsToLL_M-5to50_HT-70to100') \
                            + nanoGetSampleFiles(mcDirectory_2016, 'DYJetsToLL_M-5to50_HT-100to200') \
                            + nanoGetSampleFiles(mcDirectory_2016, 'DYJetsToLL_M-5to50_HT-200to400') \
                            + nanoGetSampleFiles(mcDirectory_2016, 'DYJetsToLL_M-5to50_HT-400to600_ext1') \
                            + nanoGetSampleFiles(mcDirectory_2016, 'DYJetsToLL_M-5to50_HT-600toinf') \
                            + nanoGetSampleFiles(mcDirectory_2016, 'DYJetsToLL_M-50_HT-70to100') \
                            + nanoGetSampleFiles(mcDirectory_2016, 'DYJetsToLL_M-50_HT-100to200_ext1') \
                            + nanoGetSampleFiles(mcDirectory_2016, 'DYJetsToLL_M-50_HT-200to400_ext1') \
                            + nanoGetSampleFiles(mcDirectory_2016, 'DYJetsToLL_M-50_HT-400to600_ext1') \
                            + nanoGetSampleFiles(mcDirectory_2016, 'DYJetsToLL_M-50_HT-600to800') \
                            + nanoGetSampleFiles(mcDirectory_2016, 'DYJetsToLL_M-50_HT-800to1200') \
                            + nanoGetSampleFiles(mcDirectory_2016, 'DYJetsToLL_M-50_HT-1200to2500') \
                            + nanoGetSampleFiles(mcDirectory_2016, 'DYJetsToLL_M-50_HT-2500toInf')

    samples['DY_2016'] = {
        'name': files,
        'weight': mcCommonWeight,
        'FilesPerJob': 8,
        'suppressNegative' :['all'],
        'suppressNegativeNuisances' :['all'],
    }
    
    addSampleWeight(samples,'DY_2016','DYJetsToLL_M-50_ext2', '(LHE_HT<70.0)')
    addSampleWeight(samples,'DY_2016','DYJetsToLL_M-10to50',  '(LHE_HT<70.0)')

    #xxxxxxx 2017 xxxxxxx
    files =     nanoGetSampleFiles(mcDirectory_2017, 'DYJetsToLL_M-50_ext1') \
                                    + nanoGetSampleFiles(mcDirectory_2017, 'DYJetsToLL_M-10to50-LO_ext1')\
                                    + nanoGetSampleFiles(mcDirectory_2017, 'DYJetsToLL_M-4to50_HT-100to200_ext1') \
                                    + nanoGetSampleFiles(mcDirectory_2017, 'DYJetsToLL_M-4to50_HT-200to400_newpmx') \
                                    + nanoGetSampleFiles(mcDirectory_2017, 'DYJetsToLL_M-4to50_HT-400to600') \
                                    + nanoGetSampleFiles(mcDirectory_2017, 'DYJetsToLL_M-4to50_HT-600toInf') \
                                    + nanoGetSampleFiles(mcDirectory_2017, 'DYJetsToLL_M-50_HT-100to200') \
                                    + nanoGetSampleFiles(mcDirectory_2017, 'DYJetsToLL_M-50_HT-200to400') \
                                    + nanoGetSampleFiles(mcDirectory_2017, 'DYJetsToLL_M-50_HT-400to600_ext1') \
                                    + nanoGetSampleFiles(mcDirectory_2017, 'DYJetsToLL_M-50_HT-600to800') \
                                    + nanoGetSampleFiles(mcDirectory_2017, 'DYJetsToLL_M-50_HT-800to1200') \
                                    + nanoGetSampleFiles(mcDirectory_2017, 'DYJetsToLL_M-50_HT-1200to2500') \
                                    + nanoGetSampleFiles(mcDirectory_2017, 'DYJetsToLL_M-50_HT-2500toInf')
    
    samples['DY_2017'] = {
        'name': files,
        'weight': mcCommonWeight,
        'FilesPerJob': 8,
        'suppressNegative' :['all'],
        'suppressNegativeNuisances' :['all'],
        }

    addSampleWeight(samples,'DY_2017','DYJetsToLL_M-50_ext1'       , '(LHE_HT<100.0)')
    addSampleWeight(samples,'DY_2017','DYJetsToLL_M-10to50-LO_ext1', '(LHE_HT<100.0)')

    

    #xxxxxxx 2018 xxxxxxx
    files =  nanoGetSampleFiles(mcDirectory_2018, 'DYJetsToLL_M-50_ext2') \
                                + nanoGetSampleFiles(mcDirectory_2018, 'DYJetsToLL_M-10to50-LO') \
                                + nanoGetSampleFiles(mcDirectory_2018, 'DYJetsToLL_M-10to50-LO_ext1') \
                                + nanoGetSampleFiles(mcDirectory_2018, 'DYJetsToLL_M-4to50_HT-100to200' ) \
                                + nanoGetSampleFiles(mcDirectory_2018, 'DYJetsToLL_M-4to50_HT-200to400' ) \
                                + nanoGetSampleFiles(mcDirectory_2018, 'DYJetsToLL_M-4to50_HT-400to600' ) \
                                + nanoGetSampleFiles(mcDirectory_2018, 'DYJetsToLL_M-4to50_HT-600toInf') \
                                + nanoGetSampleFiles(mcDirectory_2018, 'DYJetsToLL_M-50_HT-70to100') \
                                + nanoGetSampleFiles(mcDirectory_2018, 'DYJetsToLL_M-50_HT-100to200') \
                                + nanoGetSampleFiles(mcDirectory_2018, 'DYJetsToLL_M-50_HT-200to400') \
                                + nanoGetSampleFiles(mcDirectory_2018, 'DYJetsToLL_M-50_HT-400to600') \
                                + nanoGetSampleFiles(mcDirectory_2018, 'DYJetsToLL_M-50_HT-600to800') \
                                + nanoGetSampleFiles(mcDirectory_2018, 'DYJetsToLL_M-50_HT-800to1200')  \
                                + nanoGetSampleFiles(mcDirectory_2018, 'DYJetsToLL_M-50_HT-1200to2500') \
                                + nanoGetSampleFiles(mcDirectory_2018, 'DYJetsToLL_M-50_HT-2500toInf')
    samples['DY_2018'] = {
        'name': files,
        'weight': mcCommonWeight,
        'FilesPerJob': 8,
        'suppressNegative' :['all'],
        'suppressNegativeNuisances' :['all'],
    }

    addSampleWeight(samples,'DY_2018','DYJetsToLL_M-10to50-LO_ext1', '(LHE_HT<100)')
    addSampleWeight(samples,'DY_2018','DYJetsToLL_M-10to50-LO'     , '(LHE_HT<100)')
    addSampleWeight(samples,'DY_2018','DYJetsToLL_M-50_ext2'       , '(LHE_HT<70)')

# ###########################################
# #############   SIGNALS  ##################
# ###########################################

if not useDYHTWWTo2L2Nu:
    ###### WW ########
    #xxxxxxx 2016 xxxxxxx
    samples['WW_2016'] = {
        'name': nanoGetSampleFiles(mcDirectory_2016, 'WW-LO') + nanoGetSampleFiles(mcDirectory_2016, 'WW-LO_ext1'),
        'weight': mcCommonWeight,
        'FilesPerJob': 4
    }

    #xxxxxxx 2017 xxxxxxx

    samples['WW_2017'] = {
        'name': nanoGetSampleFiles(mcDirectory_2017, 'WW-LO'),
        'weight': mcCommonWeight,
        'FilesPerJob': 1
    }

    #xxxxxxx 2018 xxxxxxx

    samples['WW_2018'] = {
        'name': nanoGetSampleFiles(mcDirectory_2018, 'WW-LO'),
        'weight': mcCommonWeight,
        'FilesPerJob': 2
    }

else:
    ###### WWTo2L2Nu ########
    #xxxxxxx 2016 xxxxxxx
    samples['WW_2016'] = {
    'name': nanoGetSampleFiles(mcDirectory_2016, 'WWTo2L2Nu'),
    'weight': mcCommonWeight+'*nllW',
    'FilesPerJob': 2,
    'suppressNegative' :['all'],
    'suppressNegativeNuisances' :['all'],
    }
    #xxxxxxx 2017 xxxxxxx

    samples['WW_2017'] = {
    'name': nanoGetSampleFiles(mcDirectory_2017, 'WWTo2L2Nu'),
    'weight': mcCommonWeight+'*nllW',
    'FilesPerJob': 2,
    'suppressNegative' :['all'],
    'suppressNegativeNuisances' :['all'],
    }

    #xxxxxxx 2018 xxxxxxx

    samples['WW_2018'] = {
    'name': nanoGetSampleFiles(mcDirectory_2018, 'WWTo2L2Nu'),
    'weight': mcCommonWeight+'*nllW',
    'FilesPerJob': 2,
    'suppressNegative' :['all'],
    'suppressNegativeNuisances' :['all'],
    }