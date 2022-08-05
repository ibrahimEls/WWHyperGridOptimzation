# example of configuration file

import os

configDir = os.path.expandvars("/afs/cern.ch/user/i/ielshark/CMSSW_10_6_4/src/BDT_DY_discriminator_for_WW/Training/2jet")

tagName = ''

# luminosity to normalize to (in 1/fb)
lumi = 137.1

# file with list of cuts
cutsFile = os.path.join(configDir,'cuts_BDT.py' )

# file with list of samples
samplesFile = os.path.join(configDir,'samples_BDT.py' )

# structure file for datacard
structureFile = os.path.join(configDir,'structure.py')
