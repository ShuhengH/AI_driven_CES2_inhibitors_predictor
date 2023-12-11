import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def load_data(path):
    dataset = pd.read_csv(path,delimiter=",",header=0)
    
    suppl = [x for x in dataset["Smiles"]]
    mols = [Chem.MolFromSmiles(x) for x in suppl]
    
    # revert to canonical stru
    canonical_smi = [Chem.MolToSmiles(mol) for mol in mols]
    canonical_mols = [Chem.MolFromSmiles(x) for x in canonical_smi]
    
    return dataset,canonical_smi,canonical_mols

def calcMCFP(mols,dataset):
    MCFP = []
    for mol in mols:
        #Mkeys = list(MACCSkeys.GenMACCSKeys(mol))
        onbits = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2,nBits=128, bitInfo=onbits)
        FPkeys = list(fp)
        MCFP.append(FPkeys)
    
    data = pd.DataFrame(MCFP)
    data_concat = pd.concat([dataset["Smiles"],dataset["pIC50"], data], axis=1)
    #data_concat.to_csv("MCFP_Key.csv", index = False)
    
    return data_concat
