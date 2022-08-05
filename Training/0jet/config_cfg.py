#!/usr/bin/env python

from __future__ import print_function
import os
from ROOT import gROOT, TFile, TChain, TCut

# import models
#import preselections

isDEV=False

# Load configuration
with open("configuration.py") as handle:
    exec handle

samples={}
structure={}
cuts={}

for f in [samplesFile, structureFile, cutsFile]:
    with open(f) as handle:
        exec handle


# Reduce sample files for fast dev
for sampleName, sample in samples.items():
    if sampleName not in ['ggWW_2016','ggWW_2017','ggWW_2018','WW_2016','WW_2017','WW_2018','DY_2016','DY_2017','DY_2018']:
        samples.pop(sampleName)
        continue

    if isDEV:
        if len(sample['name']) > 2:
            sample['name'] = sample['name'][0:1]
    else :
        sample['name'] = sample['name']

# Define data to be loaded
#with open("./preselections.py") as handle:
#    exec handle

#cut="(({0}) && ({1}))".format(supercut,preselections['ALL'])
#cut="(({0}))".format(preselections['ALL'])
cut="(({0}))".format(supercut)
mvaVariables = [
    'Lepton_pt[0]',
    'Lepton_pt[1]',
   'Lepton_eta[0]',
   'Lepton_eta[1]',
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
