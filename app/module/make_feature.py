from numpy import NaN
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors 
molfile_dic = [''' 
JME 2022-02-26 Wed Jun 15 12:23:49 GMT+900 2022

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.1074    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.3959    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0  0  0  0
M  END
               ''']

def feature(molfile_dic):

    for molfile in molfile_dic.values():
        if Chem.MolFromMolBlock(molfile) is None:
            return None, None
        else:
            mol = Chem.MolFromMolBlock(molfile)
            
    # 水素追加
    mol = Chem.AddHs(mol)
    weight = rdMolDescriptors._CalcMolWt(mol)
    symbols = symobolArray(mol)
    neighbors = neighborIndexArray(mol)
    desc1 = nextLevelExpressionArray(symbols, symbols, neighbors, False)
    desc2 = nextLevelExpressionArray(symbols, desc1, neighbors);
    featureValues2 = descriptorCounts(desc2)
    fsmiles = fragmentList(mol)
    featureValuesFrag = descriptorCounts(fsmiles)
    feature = {**featureValues2, **featureValuesFrag}
    return feature, weight

    
def symobolArray(mol):
    '''
    {key = id : value = 原子} の辞書を返す
    (例) {0: 'C', 1: 'C', 2: 'H', 3: 'H', 4: 'H'...}
    '''
    symbols = {}
    for atom in mol.GetAtoms():
        id = atom.GetIdx()
        symbols[id] = atom.GetSymbol()
    return symbols

def neighborIndexArray(mol):
    '''
    {key = id : value = [隣接する原子のidのリスト]} の辞書を返す
    (例) {0: [1, 2, 3, 4], 1: [0, 5, 6, 7], 2: [0],...}
    '''
    n_atoms = len(mol.GetAtoms())
        
    neighbors = {}
    for atom in mol.GetAtoms():
        id = atom.GetIdx() 
        neighbors[id] = []
    
    for bond in mol.GetBonds():
        atom0 = bond.GetBeginAtom().GetIdx()
        atom1 = bond.GetEndAtom().GetIdx()
        
        neighbors[atom0].append(atom1)
        neighbors[atom1].append(atom0)

    return neighbors
        
def nextLevelExpressionArray(symbols, preLevelExp, neighbors, addParentheses = True):
    '''
    {key = id : value = 木構造} の辞書を返す
    
    (例) 
    addParenthese = False の場合
    {0: 'C-CHHH', 1: 'C-CHHH', 2: 'H-C',...}
    
    addParenthese = True の場合
    {0: 'C-(C-CHHH)(H-C)(H-C)(H-C)', 1: 'C-(C-CHHH)(H-C)(H-C)(H-C)', 2: 'H-(C-CHHH)',...}
    
    '''
    n = len(symbols)
    nextLevelExp = {}
    for i in range(n):
        m = len(neighbors[i])
        exp = {}
        for j in range(m):
            exp[j] = preLevelExp[neighbors[i][j]]
        nextLevelExp[i] = symbols[i] + "-";
        for j in range(m):
            if (addParentheses):
                nextLevelExp[i] += f"({exp[j]})"
            else:
                nextLevelExp[i] += exp[j]
                
    return nextLevelExp

def fragmentList(mol, max_ring=8):
    
    cut_bonds = set()
    for bond in mol.GetBonds():
        bond_id = bond.GetIdx()
        if bond.GetBondType() == 1:
            cut_bonds.add(bond_id)

    rings = mol.GetRingInfo()
    for ring in rings.BondRings():
        ring_size = len(ring)
        if ring_size <= max_ring:
            for ring_bond_id in ring:

                if ring_bond_id in cut_bonds:
                    cut_bonds.remove(ring_bond_id)
                    # print('ring8以下')

    not_cut_bonds = set()
    for bond_id in cut_bonds:
        bond = mol.GetBondWithIdx(bond_id)
        atom0 = bond.GetBeginAtom()
        atom1 = bond.GetEndAtom()
        symbol0 = atom0.GetSymbol()
        symbol1 = atom1.GetSymbol()
        number0 = atom0.GetAtomicNum()
        number1 = atom1.GetAtomicNum()
        has_bonds_count0 = len(atom0.GetBonds())
        has_bonds_count1 = len(atom1.GetBonds())
        if number0 != 6 and number1 != 6:
            not_cut_bonds.add(bond_id)
        elif has_bonds_count0 == 1 and symbol0 == "H":
            not_cut_bonds.add(bond_id)
        elif has_bonds_count1 == 1 and symbol1 == "H":
            not_cut_bonds.add(bond_id)
        elif number0 == 6 and number1 == 6:
            if has_bonds_count0 == 4 and has_bonds_count1 == 4:
                h_count0 = 0
                h_count1 = 0
                for neighbor in atom0.GetNeighbors():
                    if neighbor.GetAtomicNum() == 1:
                        h_count0 += 1
                if h_count0 == 2:
                    not_cut_bonds.add(bond_id)
                    continue
                
                for neighbor in atom1.GetNeighbors():
                    if neighbor.GetAtomicNum() == 1:
                        h_count1 += 1
                if h_count1 == 2:
                    not_cut_bonds.add(bond_id)              
                    continue
 

    cut_bonds = cut_bonds - not_cut_bonds

    # 編集ができるmol情報に変換
    edit_mol = Chem.RWMol(mol)
    
    # 水素除去
    for _ in range(len(mol.GetAtoms())):
        for atom in edit_mol.GetAtoms():
            if atom.GetSymbol() == 'H':        
                edit_mol.RemoveAtom(atom.GetIdx())
                break

    # fsmiles_has_h = []
    if cut_bonds:
        for bond_id in cut_bonds:
            bond = mol.GetBondWithIdx(bond_id)
            atom0 = bond.GetBeginAtom()
            atom1 = bond.GetEndAtom()
            id0 = atom0.GetIdx()
            id1 = atom1.GetIdx()

            # 切る
            edit_mol.RemoveBond(id0, id1)
    
    # smilesに変換 
    fragments = Chem.MolToSmiles(edit_mol)
    
    # 辞書に変換
    fsmiles = fragments.split('.')
    fsmiles = {k:smiles for k, smiles in enumerate(fsmiles)}

    return fsmiles



def descriptorCounts(descriptors):
    '''
    {key = 記述子 : value = 記述子の個数} の辞書を返す
    (例) {'C-(C-CHHH)(H-C)(H-C)(H-C)': 2, 'H-(C-CHHH)': 6}
    '''
    counts = {};
    for d in descriptors.values():
        if d in counts:
            counts[d] += 1
        else:
            counts[d] = 1;
    return counts

if __name__ == "__main__": 
  feature(molfile_dic)
# use matsuiLab
#     for molfile in molfile_dic:
#         mol = chem.FlexMol(molfile)

#     # 水素追加  
#     mol.add_hydrogens()
    
#     symbols = symobolArray(mol)
#     neighbors = neighborIndexArray(mol)
#     desc1 = nextLevelExpressionArray(symbols, symbols, neighbors, False)
#     desc2 = nextLevelExpressionArray(symbols, desc1, neighbors);
#     featureValues2 = descriptorCounts(desc2)
#     print(featureValues2)

    
#     fsmiles = fragmentList(mol, max_ring=8)
    # featureValuesFrag = descriptorCounts(fsmiles)
    # print(featureValuesFrag)

    
# def symobolArray(mol):
#     '''
#     {key = id : value = 原子} の辞書を返す
#     (例) {0: 'C', 1: 'C', 2: 'H', 3: 'H', 4: 'H'...}
#     '''
#     symbols = {}
#     for i, atom in enumerate(mol.atoms):
#         symbols[i] = atom.symbol
#     return symbols

# def neighborIndexArray(mol):
#     '''
#     {key = id : value = [隣接する原子のidのリスト]} の辞書を返す
#     (例) {0: [1, 2, 3, 4], 1: [0, 5, 6, 7], 2: [0],...}
#     '''
#     n_atoms = len(mol.atoms)
#     for i in range(n_atoms):
#         mol.atoms[i].id = i
        
#     neighbors = {}
#     for i in range(n_atoms):
#         neighbors[i] = []
        
#     for conn in mol.bonds:
#         objs = conn.atoms
#         atom0 = objs[0].id
#         atom1 = objs[1].id
        
#         neighbors[atom0].append(atom1)
#         neighbors[atom1].append(atom0)
        
#     return neighbors
        
# def nextLevelExpressionArray(symbols, preLevelExp, neighbors, addParentheses = True):
#     '''
#     {key = id : value = 木構造} の辞書を返す
    
#     (例) 
#     addParenthese = False の場合
#     {0: 'C-CHHH', 1: 'C-CHHH', 2: 'H-C',...}
    
#     addParenthese = True の場合
#     {0: 'C-(C-CHHH)(H-C)(H-C)(H-C)', 1: 'C-(C-CHHH)(H-C)(H-C)(H-C)', 2: 'H-(C-CHHH)',...}
    
#     '''
#     n = len(symbols)
#     nextLevelExp = {}
#     for i in range(n):
#         m = len(neighbors[i])
#         exp = {}
#         for j in range(m):
#             exp[j] = preLevelExp[neighbors[i][j]]
#         nextLevelExp[i] = symbols[i] + "-";
#         for j in range(m):
#             if (addParentheses):
#                 nextLevelExp[i] += f"({exp[j]})"
#             else:
#                 nextLevelExp[i] += exp[j]
                
#     return nextLevelExp

# def descriptorCounts(descriptors):
#     '''
#     {key = 記述子 : value = 記述子の個数} の辞書を返す
#     (例) {'C-(C-CHHH)(H-C)(H-C)(H-C)': 2, 'H-(C-CHHH)': 6}
#     '''
#     counts = {};
#     for d in descriptors.values():
#         if d in counts:
#             counts[d] += 1
#         else:
#             counts[d] = 1;
#     return counts