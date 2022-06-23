import numpy as np
from rdkit import Chem
import pickle
from module.mol_info import MolInfo

# with open("../models/ridgeCV_descriptors_dic_rdkit.sav", "rb") as f:
#     ml_feature = pickle.load(f)
# molfile_dic = {'molfile':''' 
# JME 2022-02-26 Wed Jun 15 12:23:49 GMT+900 2022

#   2  1  0  0  0  0  0  0  0  0999 V2000
#     0.0000    0.1074    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
#     1.3959    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
#   1  2  2  0  0  0  0
# M  END
#                '''}


class RidgeOrb(MolInfo):
    
    def makeFeature(self):
        symbols = self.symobolArray()
        neighbors = self.neighborIndexArray()
        desc1 = self.nextLevelExpressionArray(symbols, symbols, neighbors, False)
        desc2 = self.nextLevelExpressionArray(symbols, desc1, neighbors);
        featureValues2 = self.descriptorCounts(desc2)
        fsmiles = self.fragmentList()
        featureValuesFrag = self.descriptorCounts(fsmiles)
        js_feature = {**featureValues2, **featureValuesFrag}
        return js_feature
        
    def symobolArray(self):
        '''
        {key = id : value = 原子} の辞書を返す
        (例) {0: 'C', 1: 'C', 2: 'H', 3: 'H', 4: 'H'...}
        '''
        symbols = {}
        for atom in self.mol.GetAtoms():
            id = atom.GetIdx()
            symbols[id] = atom.GetSymbol()
        return symbols

    def neighborIndexArray(self):
        '''
        {key = id : value = [隣接する原子のidのリスト]} の辞書を返す
        (例) {0: [1, 2, 3, 4], 1: [0, 5, 6, 7], 2: [0],...}
        '''
            
        neighbors = {}
        for atom in self.mol.GetAtoms():
            id = atom.GetIdx() 
            neighbors[id] = []
        
        for bond in self.mol.GetBonds():
            atom0 = bond.GetBeginAtom().GetIdx()
            atom1 = bond.GetEndAtom().GetIdx()
            
            neighbors[atom0].append(atom1)
            neighbors[atom1].append(atom0)

        return neighbors
            
    def nextLevelExpressionArray(self, symbols, preLevelExp, neighbors, addParentheses = True):
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

    def fragmentList(self, max_ring=8):
        
        cut_bonds = set()
        for bond in self.mol.GetBonds():
            bond_id = bond.GetIdx()
            if bond.GetBondType() == 1:
                cut_bonds.add(bond_id)

        rings = self.mol.GetRingInfo()
        for ring in rings.BondRings():
            ring_size = len(ring)
            if ring_size <= max_ring:
                for ring_bond_id in ring:

                    if ring_bond_id in cut_bonds:
                        cut_bonds.remove(ring_bond_id)
                        # print('ring8以下')

        not_cut_bonds = set()
        for bond_id in cut_bonds:
            bond = self.mol.GetBondWithIdx(bond_id)
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
        edit_mol = Chem.RWMol(self.mol)
        
        edit_mol = self.removeHs(edit_mol)
        
        if cut_bonds:
            for bond_id in cut_bonds:
                bond = self.mol.GetBondWithIdx(bond_id)
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

    def descriptorCounts(self, descriptors):
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
    
    def removeHs(self, edit_mol):
        for _ in range(len(self.mol.GetAtoms())):
            for atom in edit_mol.GetAtoms():
                if atom.GetSymbol() == 'H':        
                    edit_mol.RemoveAtom(atom.GetIdx())
                    break
        return edit_mol
    
    def predict(self, js_feature, ml_feature):
        '''
        js_feature = keyがYU_canvasから入力された分子構造の記述子、valueが記述子の個数のdic
        ml_feature = keyが機械学習モデルの記述子、valueが個数がすべて0のdic
        return = homoとlumoの予測値をdicで返している
        '''
        
        # 重みでモデルを変更
        if self.weight <= 287:
            weight_div = 0
        elif self.weight <= 369:
            weight_div = 1
        elif self.weight <= 486:
            weight_div = 2
        else:
            weight_div = 3
            
        # モデルを読み込む
        with open(f"/app/models/ridge/4div_{weight_div}_ridgeCV_fragment_depth2_all_rdkit_HOMO.sav", "rb") as f:
            model_homo = pickle.load(f)
        with open(f"/app/models/ridge/4div_{weight_div}_ridgeCV_fragment_depth2_all_rdkit_LUMO.sav", "rb") as f:    
            model_lumo = pickle.load(f)
        
        match_count = 0
        for js_f in js_feature:   
        # ml_featureにjs_featureの個数を追加
            if js_f in ml_feature:
                ml_feature[js_f] = js_feature[js_f]
                match_count += 1
                print(f'matching: {js_f}')
            else:
                print(f'No match: {js_f}')
        rate = (match_count / len(js_feature)) * 100
        print(rate)
        
                
        # 個数のみの配列を作成し、モデルに挿入
        # descript_dic1 = {**descript_dic, **feature_dic}
        # if len(descript_dic1) != len(descript_dic):
        #     pre_orb = {'homo': 0, 'lumo': 0}
        #     return pre_orb

        descript_count = list(ml_feature.values())
        descript_array = np.array([descript_count], dtype=int)

        # 予測値をdicに格納
        homo = model_homo.predict(descript_array)[0]
        lumo = model_lumo.predict(descript_array)[0]
        ridge_predict_orb = {'homo': homo, 'lumo': lumo, 'rate': rate}
        return ridge_predict_orb
        

    

    
# ridge_orb = RidgeOrb(molfile_dic)
# js_feature = ridge_orb.makeFeature()
# ridge_predict_orb = ridge_orb.predict(js_feature, ml_feature)
# print(ridge_predict_orb)

        
        
        