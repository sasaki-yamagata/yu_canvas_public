from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

class MolInfo(object):
    
    def __init__(self, molfile_dic):
        
        for molfile in molfile_dic.values():
            if Chem.MolFromMolBlock(molfile) is None:
                return None
        else:
            mol = Chem.MolFromMolBlock(molfile)
            
        self.smiles = Chem.MolToSmiles(mol)
        mol = Chem.AddHs(mol)
        self.mol = mol
        self.weight = rdMolDescriptors._CalcMolWt(mol)