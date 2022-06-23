# coding: utf-8
from common import config
# GPUで実行する場合は、下記のコメントアウトを消去（要cupy）
# ===============================================
# config.GPU = True
# ===============================================
from common.optimizer import *
from common.multi_layer_net_extend import MultiLayerNetExtend
from rdkit import Chem
import pickle
from common.np import *
from module.mol_info import MolInfo

# sys.path.append(os.pardir)
# print = functools.partial(print, flush=True)
# os.chdir(os.path.dirname(__file__))

class GcnOrb(MolInfo):
    def predict(self):
         # シード値の指定（特にいらないかも）
        np.random.seed(0)

        # 特徴量行列を作成するのに必要
        addHs = True
        if addHs:
            # 水素有り
            input_size = 22
            symbol_list = ['As', 'B', 'Br', 'C', 'Cl', 'F', 'H', 'Mg', 'N', 'O', 'P', 'S', 'Se', 'Si']
            hybrid_list = [Chem.rdchem.HybridizationType(0), Chem.rdchem.HybridizationType(2), 
                                    Chem.rdchem.HybridizationType(3), Chem.rdchem.HybridizationType(4), 
                                    Chem.rdchem.HybridizationType(5), Chem.rdchem.HybridizationType(6)]
        else:
            # 水素なし
            input_size = 20
            symbol_list = ['As', 'B', 'Br', 'C', 'Cl', 'F', 'Mg', 'N', 'O', 'P', 'S', 'Se', 'Si']
            hybrid_list = [Chem.rdchem.HybridizationType(2), Chem.rdchem.HybridizationType(3), 
                                    Chem.rdchem.HybridizationType(4), Chem.rdchem.HybridizationType(5), 
                                    Chem.rdchem.HybridizationType(6)]

        # 学習時の層数やノード数の設定
        bayes_list = [3, 1, 6]
        gc_hidden_size_list = [
            2**int(bayes_list[2]) if _ % 2 == 0 else 2**int(bayes_list[2])*2 for _ in range(int(bayes_list[0]))]
        affine_hidden_size_list = [
            2**int(bayes_list[2])*2 if _ % 2 == 0 else 2**int(bayes_list[2]) for _ in range(int(bayes_list[1]))] + [4]

        # networkの初期構築
        network_HOMO = MultiLayerNetExtend(
            input_size, gc_hidden_size_list, affine_hidden_size_list, 1, pooling_list=False, 
            sim_global_attention_list=False, loss_function='MSE',dis_global_attention_list=False, 
            gc_global=False, weight_decay_lambda_affine=0, weight_decay_lambda_gc=0,
            gg='average', activation='tanhEXP', weight_init_std='glorot', gc_bias=True, gc_double_W=True, 
            use_batchnorm=False, use_dropout=False, dropout_ration=0.5)
        network_LUMO = MultiLayerNetExtend(
            input_size, gc_hidden_size_list, affine_hidden_size_list, 1, pooling_list=False, 
            sim_global_attention_list=False, loss_function='MSE',dis_global_attention_list=False, 
            gc_global=False, weight_decay_lambda_affine=0, weight_decay_lambda_gc=0,
            gg='average', activation='tanhEXP', weight_init_std='glorot', gc_bias=True, gc_double_W=True, 
            use_batchnorm=False, use_dropout=False, dropout_ration=0.5)

        # 学習した重みデータを読み込み、networkに入れる。
        with open('/app/models/gcn/gcn_weight_HOMO_addHs_norm_limit_scaler_bayescond_3,1,6,4,-350,-350,-1.pkl', 'rb') as binary_reader:
            params_dict_numpy = pickle.load(binary_reader)
            params_dict = {}
            for key in params_dict_numpy.keys():
                params_dict[key] = np.array(params_dict_numpy[key])
            network_HOMO.set_params(params_dict)

        with open('/app/models/gcn/gcn_weight_LUMO_addHs_norm_limit_scaler_bayescond_3,1,6,4,-350,-350,-1.pkl', 'rb') as binary_reader:
            params_dict_numpy = pickle.load(binary_reader)
            params_dict = {}
            for key in params_dict_numpy.keys():
                params_dict[key] = np.array(params_dict_numpy[key])
            network_LUMO.set_params(params_dict)
        
        # 特徴量行列と隣接行列をSMILESから作成
        f_matrix, adj_matrix = self.make_feature(symbol_list, hybrid_list, addHs)

        # 値を予測する。
        # f_matrixがNoneだったら（SMILESからRDKitのmol型に変換できなかったら）エラーを返す。
        if f_matrix is None:
            raise ValueError('ERROR!:分子に変換できませんでした。')
        # 問題なければ予測し、出力
        else: 
            homo = '{:6.2f}'.format(network_HOMO.predict(f_matrix, adj_matrix)[0,0])
            lumo = '{:6.2f}'.format(network_LUMO.predict(f_matrix, adj_matrix)[0,0])
            gcn_predict_orb = {'homo' : homo, 'lumo' : lumo}
            return gcn_predict_orb
        
    def make_feature(self, symbol_list, hybrid_list, add_Hs=False):
        '''
        グラフ畳み込みに使用する行列を出力する。

        Argument:
            smiles {str}        -- SMILESの文字列
            symbol_list {list}  -- モデルに対応している元素記号のリスト
            hybrid_list {list}  -- モデルに対応している混成軌道のリスト
            add_Hs {bool}       -- 水素を含めるかどうか。デフォルトはFalse（水素を含めない）。
            
        Returns:
            f_matrixs {array}   -- 特徴量ベクトルを集めて行列にしたもの。（特徴量行列）
            adj_matrix {array}  -- 隣接行列
        '''
        # SMILESからRDKitのmol型に変換
        mol = Chem.MolFromSmiles(self.smiles)
        # molがNoneだったら（mol型に変換できなかったら）Noneを戻り値とする。
        if mol is None:
            return None, None
        # 水素を追加する場合ここで追加。
        if add_Hs:
            mol = Chem.rdmolops.AddHs(mol)
        # 原子情報の取得
        atoms = mol.GetAtoms()

        # 特徴量行列の大枠を作成
        f_matrix = np.zeros((mol.GetNumAtoms(), len(symbol_list)+len(hybrid_list)+2))
        # f_matrixに特徴量を入れていく
        count = 0
        for atom in atoms:
            # 元素記号のone-hot表現
            f_matrix[count, symbol_list.index(atom.GetSymbol())] += 1
            # 総原子価
            f_matrix[count, len(symbol_list)] += atom.GetTotalValence()
            # 形式電荷
            f_matrix[count, len(symbol_list)+1] += atom.GetFormalCharge()
            # 混成軌道の種類のone-hot表現
            f_matrix[count, hybrid_list.index(
                atom.GetHybridization())+len(symbol_list)+2] += 1
            count += 1

        # 隣接行列の取得
        adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
        # 隣接行列の正規化
        adj_matrix = self.AdjNormalization(adj_matrix)

        return f_matrix.reshape(1,f_matrix.shape[0], f_matrix.shape[1]), adj_matrix.reshape(1,adj_matrix.shape[0], adj_matrix.shape[1])


    def AdjNormalization(self, adj_matrix):
        '''
        隣接行列を正規化する。（正規化グラフラプラシアンにする）
        参照: https://www.slideshare.net/ryosuke-kojima/ss-179423718 の55枚目のスライド
        '''
        # 次数行列の作成
        dim = np.zeros(adj_matrix.shape)
        adj_sum = adj_matrix.sum(axis=1)
        for i in range(adj_matrix.shape[0]):
            dim[i, i] = adj_sum[i]
        
        # 正規化グラフラプラシアンの作成
        std_adj = np.sqrt(np.linalg.inv(dim))@(dim-adj_matrix)@np.sqrt(np.linalg.inv(dim))
        # std_adj1 = np.empty(adj_matrix.shape, np.float64)
        # for i in range(adj_matrix.shape[0]):
        #     for j in range(adj_matrix.shape[1]):
        #         if i == j and dim[i, j] != 0:
        #             std_adj1[i, j] = 1
        #         elif i != j and adj_matrix[i, j] != 0:
        #             std_adj1[i, j] = -adj_matrix[i, j] / \
        #                 (np.sqrt(dim[i, i]*dim[j, j]))
        #         else:
        #             std_adj1[i, j] = 0
        # np.set_printoptions(precision=2)
        # print(std_adj1)
        # print((std_adj1-std_adj).sum())
        # print(std_adj)
        return std_adj

    
