# cuts
supercut = '   mll>20 \
            && Lepton_pt[0]>25 \
            && Lepton_pt[1]>20 \
            && (fabs(91.1876 - mll) > 15) \
            && (abs(Lepton_pdgId[1])==13 || Lepton_pt[1]>13) \
            && (nLepton>=2 && Alt$(Lepton_pt[2],0)<10) \
            && abs(Lepton_eta[0])<2.5 && abs(Lepton_eta[1])<2.5 \
            && ptll>30 \
            && PuppiMET_pt > 20 \
            && ((Lepton_pdgId[0]*Lepton_pdgId[1] == -11*11) || (Lepton_pdgId[0]*Lepton_pdgId[1] == -13*13)) \
            && Alt$(CleanJet_pt[0], 0) < 30. \
               '
