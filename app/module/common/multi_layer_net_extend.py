# coding: utf-8
import functools
from networkx.classes.function import freeze
from common.gradient import numerical_gradient
from common.layers import *
from collections import OrderedDict
from common.np import *
import sys
import os
import time
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
# np.random.seed(0)
print = functools.partial(print, flush=True)


class MultiLayerNetExtend:
    """グラフ畳み込みによる多層ニューラルネットワーク

    Weiht Decay、Dropout、Batch Normalizationの機能を持つ

    Parameters
    ----------
    input_size : 入力サイズ（MNISTの場合は784）
    gc_hidden_size_list : グラフ畳み込み隠れ層のニューロンの数のリスト（e.g. [100, 100, 100]）
    affine_hidden_size_list : アフィン隠れ層のニューロンの数のリスト（e.g. [100, 100, 100]）
    output_size : 出力サイズ（MNISTの場合は10）
    loss_function : 損失関数 'MSE' or 'MSErange'
    pooling_list : GraphPooling層を入れるかどうかのリスト（e.g. [True, True, True, False]）
        リストではなくTrueのみ入れた場合、全てにPooling層を設定
                     Falseのみ入れた場合、Pooling層は設定しない。
    activation : 'relu' or 'sigmoid' or 'swish' or 'mish' or 'absolute_value' or 'leaky_relu' or smooth_absolute_value
    weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
        'relu'または'he'を指定した場合は「Heの初期値」を設定
        'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
        'glorot'を指定した場合は「Glorotの初期値」を設定
    weight_decay_lambda_affine : affine層のWeight Decay（L2ノルム）の強さ
    weight_decay_lambda_gc : グラフ畳み込み層のWeight Decay（L2ノルム）の強さ
    use_dropout: Dropoutを使用するかどうか
    dropout_ration : Dropoutの割り合い
    use_batchNorm: Batch Normalizationを使用するかどうか
    features_merge: 集約後に別の特徴量を追加するかどうか
    merge_size: merge層を入れる際に何個の特徴量を追加するか
    """

    def __init__(self, input_size, gc_hidden_size_list, affine_hidden_size_list, output_size, loss_function='MSE',
                 pooling_list=False, sim_global_attention_list=False, dis_global_attention_list=False, gc_global=False,
                 freeze_list=False, gg='average', activation='tanhEXP', weight_init_std='glorot', gc_bias=True, gc_double_W=True,
                 weight_decay_lambda_affine=0, weight_decay_lambda_gc=0, use_dropout=False, dropout_ration=0.5, use_batchnorm=False, 
                 GC_batch_norm_size=0, features_merge=False, merge_size=0):
        self.input_size = input_size
        self.output_size = output_size
        self.gc_hidden_size_list = gc_hidden_size_list
        self.affine_hidden_size_list = affine_hidden_size_list
        self.gc_hidden_layer_num = len(gc_hidden_size_list)
        self.affine_hidden_layer_num = len(affine_hidden_size_list)
        self.gc_bias = gc_bias
        self.gc_double_W = gc_double_W
        self.use_dropout = use_dropout
        self.weight_decay_lambda_affine = weight_decay_lambda_affine
        self.weight_decay_lambda_gc = weight_decay_lambda_gc
        self.use_batchnorm = use_batchnorm
        self.gc_global = gc_global
        self.features_merge = features_merge
        self.merge_size = merge_size
        self.params = {}

        # 類似度によるGlobalAttention層に関して
        if isinstance(sim_global_attention_list, bool):
            if sim_global_attention_list:
                sim_global_attention_list = [
                    True for x in range(self.gc_hidden_layer_num)]
            else:
                sim_global_attention_list = [
                    False for x in range(self.gc_hidden_layer_num)]

        # 類似度によるGlobalAttention層に関して
        if isinstance(dis_global_attention_list, bool):
            if dis_global_attention_list:
                dis_global_attention_list = [
                    True for x in range(self.gc_hidden_layer_num)]
            else:
                dis_global_attention_list = [
                    False for x in range(self.gc_hidden_layer_num)]

        # GraphPooling層に関して
        if isinstance(pooling_list, bool):
            if pooling_list:
                pooling_list = [True for x in range(self.gc_hidden_layer_num)]
            else:
                pooling_list = [False for x in range(self.gc_hidden_layer_num)]

        # 重みの固定に関して
        if isinstance(freeze_list, bool):
            if freeze_list:
                freeze_list = [True for x in range(
                    self.gc_hidden_layer_num+self.affine_hidden_layer_num)]
            else:
                freeze_list = [False for x in range(
                    self.gc_hidden_layer_num+self.affine_hidden_layer_num)]

        # 重みの初期化
        self.__init_weight(weight_init_std)

        # レイヤの生成
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu, 'swish': Swish, 'leaky_relu': LeakyRelu,
                            'absolute_value': AbsoluteValue, 'smooth_absolute_value': SmoothAbsoluteValue,
                            'mish': Mish, 'tanhexp': TanhExp, 'softmax': Softmax}
        self.layers = OrderedDict()
        for idx in range(1, self.gc_hidden_layer_num+1):
            if freeze_list[idx-1]:
                if gc_global:
                    if not gc_bias and not gc_double_W:
                        self.layers['GlobalGraphConvolution' +
                                    str(idx)] = GlobalGraphConvolution(W=self.params['W' + str(idx)], freeze=True)
                    elif gc_bias and not gc_double_W:
                        self.layers['GlobalGraphConvolution' + str(idx)] = GlobalGraphConvolution(
                            W=self.params['W' + str(idx)], b=self.params['b' + str(idx)], freeze=True)
                    elif not gc_bias and gc_double_W:
                        self.layers['GlobalGraphConvolution' + str(idx)] = GlobalGraphConvolution(
                            W=self.params['W' + str(idx)], U=self.params['U' + str(idx)], freeze=True)
                    elif gc_bias and gc_double_W:
                        self.layers['GlobalGraphConvolution' + str(idx)] = GlobalGraphConvolution(
                            W=self.params['W' + str(idx)], U=self.params['U' + str(idx)], b=self.params['b' + str(idx)], freeze=True)

                else:
                    if not gc_bias and not gc_double_W:
                        self.layers['GraphConvolution' +
                                    str(idx)] = GraphConvolution(W=self.params['W' + str(idx)], freeze=True)
                    elif gc_bias and not gc_double_W:
                        self.layers['GraphConvolution' + str(idx)] = GraphConvolution(
                            W=self.params['W' + str(idx)], b=self.params['b' + str(idx)], freeze=True)
                    elif not gc_bias and gc_double_W:
                        self.layers['GraphConvolution' + str(idx)] = GraphConvolution(
                            W=self.params['W' + str(idx)], U=self.params['U' + str(idx)], freeze=True)
                    elif gc_bias and gc_double_W:
                        self.layers['GraphConvolution' + str(idx)] = GraphConvolution(
                            W=self.params['W' + str(idx)], U=self.params['U' + str(idx)], b=self.params['b' + str(idx)], freeze=True)

            else:
                if gc_global:
                    if not gc_bias and not gc_double_W:
                        self.layers['GlobalGraphConvolution' +
                                    str(idx)] = GlobalGraphConvolution(W=self.params['W' + str(idx)])
                    elif gc_bias and not gc_double_W:
                        self.layers['GlobalGraphConvolution' + str(idx)] = GlobalGraphConvolution(
                            W=self.params['W' + str(idx)], b=self.params['b' + str(idx)])
                    elif not gc_bias and gc_double_W:
                        self.layers['GlobalGraphConvolution' + str(idx)] = GlobalGraphConvolution(
                            W=self.params['W' + str(idx)], U=self.params['U' + str(idx)])
                    elif gc_bias and gc_double_W:
                        self.layers['GlobalGraphConvolution' + str(idx)] = GlobalGraphConvolution(
                            W=self.params['W' + str(idx)], U=self.params['U' + str(idx)], b=self.params['b' + str(idx)])

                else:
                    if not gc_bias and not gc_double_W:
                        self.layers['GraphConvolution' +
                                    str(idx)] = GraphConvolution(W=self.params['W' + str(idx)])
                    elif gc_bias and not gc_double_W:
                        self.layers['GraphConvolution' + str(idx)] = GraphConvolution(
                            W=self.params['W' + str(idx)], b=self.params['b' + str(idx)])
                    elif not gc_bias and gc_double_W:
                        self.layers['GraphConvolution' + str(idx)] = GraphConvolution(
                            W=self.params['W' + str(idx)], U=self.params['U' + str(idx)])
                    elif gc_bias and gc_double_W:
                        self.layers['GraphConvolution' + str(idx)] = GraphConvolution(
                            W=self.params['W' + str(idx)], U=self.params['U' + str(idx)], b=self.params['b' + str(idx)])

            if self.use_batchnorm:
                self.params['GC_gamma' +
                            str(idx)] = np.ones(gc_hidden_size_list[idx-1] * GC_batch_norm_size)
                self.params['GC_beta' +
                            str(idx)] = np.zeros(gc_hidden_size_list[idx-1] * GC_batch_norm_size)
                self.layers['GC_BatchNorm' + str(idx)] = BatchNormalization(
                    self.params['GC_gamma' + str(idx)], self.params['GC_beta' + str(idx)])

            self.layers[f'GC_{activation}' +
                        str(idx)] = activation_layer[activation.lower()]()

            if sim_global_attention_list[idx-1]:
                self.layers['SimilalityGlobalAttention' +
                            str(idx)] = SimilalityGlobalAttention()

            if dis_global_attention_list[idx-1]:
                self.layers['DistanceGlobalAttention' +
                            str(idx)] = DistanceGlobalAttention()

            if pooling_list[idx-1]:
                self.layers['GraphPooling' + str(idx)] = GraphPooling()

            if self.use_dropout:
                self.layers['GC_Dropout' + str(idx)] = Dropout(dropout_ration)

        if gg.lower() == 'average':
            self.layers['GraphGathering_norm'] = GraphGathering_norm()
        elif gg.lower() == 'sum':
            self.layers['GraphGathering'] = GraphGathering()
        elif gg.lower() == 'softmax':
            self.layers['GraphGathering_softmax'] = GraphGathering_softmax()

        if features_merge:
            self.layers['Merge'] = Merge() 

        for idx in range(1, self.affine_hidden_layer_num+1):
            if freeze_list[self.gc_hidden_layer_num+idx-1]:
                self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(
                    idx + self.gc_hidden_layer_num)], self.params['b' + str(idx + self.gc_hidden_layer_num)], freeze=True)
            else:
                self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(
                    idx + self.gc_hidden_layer_num)], self.params['b' + str(idx + self.gc_hidden_layer_num)])

            if self.use_batchnorm:
                self.params['Affine_gamma' +
                            str(idx)] = np.ones(affine_hidden_size_list[idx-1])
                self.params['Affine_beta' +
                            str(idx)] = np.zeros(affine_hidden_size_list[idx-1])
                self.layers['Affine_BatchNorm' + str(idx)] = BatchNormalization(
                    self.params['Affine_gamma' + str(idx)], self.params['Affine_beta' + str(idx)])

            self.layers[f'Affine_{activation}' +
                        str(idx)] = activation_layer[activation.lower()]()

            if self.use_dropout:
                self.layers['Affine_Dropout' +
                            str(idx)] = Dropout(dropout_ration)

        idx = self.affine_hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(
            self.params['W' + str(idx + self.gc_hidden_layer_num)], self.params['b' + str(idx + self.gc_hidden_layer_num)])

        if loss_function.lower() == 'mse':
            self.last_layer = MeanSquaredError()
        elif loss_function.lower() == 'mserange':
            self.last_layer = MeanSquaredOutRangeError()
        else:
            print('Error')

    def __init_weight(self, weight_init_std):
        """重みの初期値設定

        Parameters
        ----------
        weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
            'relu'または'he'を指定した場合は「Heの初期値」を設定
            'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
        """
        all_size_list = [self.input_size] + self.gc_hidden_size_list + \
            self.affine_hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                # ReLUを使う場合に推奨される初期値
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                # sigmoidを使う場合に推奨される初期値
                scale = np.sqrt(1.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ('glorot'):
                scale = np.sqrt(
                    2.0 / (all_size_list[idx - 1] + all_size_list[idx]))
            if idx == (self.gc_hidden_layer_num+1) and self.features_merge:
                self.params['W' + str(idx)] = scale * \
                    np.random.randn(all_size_list[idx-1]+self.merge_size, all_size_list[idx])
            else:
                self.params['W' + str(idx)] = scale * \
                    np.random.randn(all_size_list[idx-1], all_size_list[idx])
            if idx <= self.gc_hidden_layer_num and self.gc_double_W:
                self.params['U' + str(idx)] = scale * \
                    np.random.randn(all_size_list[idx-1], all_size_list[idx])
            if idx > self.gc_hidden_layer_num or self.gc_bias:
                self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x, adjacency_matrix, adj_shortest_matrix=None, another_features=None, train_flg=False):
        # print('------------------------------------')
        for key, layer in self.layers.items():
            # start = time.time()
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            elif "GraphConvolution" in key or "GraphPooling" in key or "GraphGathering_norm" in key or "DistanceGlobalAttention" in key or "GlobalGraphConvolution" in key:
                x = layer.forward(x, adjacency_matrix)
            elif "DistanceGlobalAttention" in key:
                x = layer.forward(x, adj_shortest_matrix)
            elif "Merge" in key:
                x = layer.forward(x, another_features)
            else:
                x = layer.forward(x)
            # end = time.time()
            # print(key,':')
            # print(end-start)
            # print('------------------------------------')

        return x

    def loss(self, x, adjacency_matrix, t, adj_shortest_matrix=None, another_features=None, train_flg=False):
        """損失関数を求める
        引数のxは入力データ、tは教師ラベル
        """
        y = self.predict(x, adjacency_matrix,adj_shortest_matrix=adj_shortest_matrix, another_features=another_features, train_flg=train_flg)

        weight_decay = 0
        for idx in range(1, self.gc_hidden_layer_num + 1):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda_gc * np.sum(W**2)
            if self.gc_double_W:
                U = self.params['U' + str(idx)]
                weight_decay += 0.5 * \
                    self.weight_decay_lambda_gc * np.sum(U**2)
                # weight_decay /= 2

        for idx in range(1, self.affine_hidden_layer_num + 2):
            W = self.params['W' + str(idx + self.gc_hidden_layer_num)]
            weight_decay += 0.5 * \
                self.weight_decay_lambda_affine * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, adjacency_matrix, t, adj_shortest_matrix=None):
        # 精度の評価
        y = self.predict(x, adjacency_matrix, adj_shortest_matrix)
        y = y.reshape(1, -1)
        t = t.reshape(1, -1)
        accuracy = np.sum(np.abs(y-t))/(y.shape[1])
        return accuracy

    def accuracy_sum(self, x, adjacency_matrix, t, adj_shortest_matrix=None):
        # 精度の評価
        y = self.predict(x, adjacency_matrix, adj_shortest_matrix)
        y = y.reshape(1, -1)
        t = t.reshape(1, -1)
        accuracy = np.sum(np.abs(y-t))
        return accuracy

    def numerical_gradient(self, x, a, t, adj_shortest_matrix=None, another_features=None):
        """勾配を求める（数値微分）

        Parameters
        ----------
        x : 入力データ
        a : 隣接行列
        t : 教師ラベル

        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        def loss_W(W): return self.loss(
            x, a, t, adj_shortest_matrix=adj_shortest_matrix, another_features=another_features, train_flg=True)

        grads = {}
        for idx in range(1, self.gc_hidden_layer_num + 1):
            grads['W' + str(idx)] = numerical_gradient(loss_W,
                                                       self.params['W' + str(idx)])
            if self.gc_double_W:
                grads['U' + str(idx)] = numerical_gradient(loss_W,
                                                           self.params['U' + str(idx)])
            if self.gc_bias:
                grads['b' + str(idx)] = numerical_gradient(loss_W,
                                                           self.params['b' + str(idx)])
            if self.use_batchnorm:
                grads['GC_gamma' + str(idx)] = numerical_gradient(loss_W,
                                                                  self.params['GC_gamma' + str(idx)])
                grads['GC_beta' + str(idx)] = numerical_gradient(loss_W,
                                                                 self.params['GC_beta' + str(idx)])

        for idx in range(1, self.affine_hidden_layer_num + 2):
            grads['W' + str(idx + self.gc_hidden_layer_num)] = numerical_gradient(loss_W,
                                                                                  self.params['W' + str(idx + self.gc_hidden_layer_num)])
            grads['b' + str(idx + self.gc_hidden_layer_num)] = numerical_gradient(loss_W,
                                                                                  self.params['b' + str(idx + self.gc_hidden_layer_num)])

            if self.use_batchnorm and idx != self.affine_hidden_layer_num+1:
                grads['Affine_gamma' + str(idx)] = numerical_gradient(loss_W,
                                                                      self.params['Affine_gamma' + str(idx)])
                grads['Affine_beta' + str(idx)] = numerical_gradient(loss_W,
                                                                     self.params['Affine_beta' + str(idx)])

        return grads

    def gradient(self, x, a, t, adj_shortest_matrix=None, another_features=None):
        # forward
        self.loss(x, a, t, adj_shortest_matrix=adj_shortest_matrix,
                  another_features=another_features, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            print(layer)
            print(dout.shape)

        # 設定
        grads = {}
        for idx in range(1, self.gc_hidden_layer_num + 1):
            if self.gc_global:
                grads['W' + str(idx)] = self.layers['GlobalGraphConvolution' + str(idx)].dW + \
                    self.weight_decay_lambda_gc * self.params['W' + str(idx)]
                if self.gc_double_W:
                    grads['U' + str(idx)] = self.layers['GlobalGraphConvolution' + str(idx)].dU + \
                        self.weight_decay_lambda_gc * \
                        self.params['U' + str(idx)]
                if self.gc_bias:
                    grads['b' + str(idx)
                          ] = self.layers['GlobalGraphConvolution' + str(idx)].db
            else:
                grads['W' + str(idx)] = self.layers['GraphConvolution' + str(idx)].dW + \
                    self.weight_decay_lambda_gc * self.params['W' + str(idx)]
                if self.gc_double_W:
                    grads['U' + str(idx)] = self.layers['GraphConvolution' + str(idx)].dU + \
                        self.weight_decay_lambda_gc * \
                        self.params['U' + str(idx)]
                if self.gc_bias:
                    grads['b' + str(idx)
                          ] = self.layers['GraphConvolution' + str(idx)].db

            if self.use_batchnorm:
                grads['GC_gamma' +
                      str(idx)] = self.layers['GC_BatchNorm' + str(idx)].dgamma
                grads['GC_beta' +
                      str(idx)] = self.layers['GC_BatchNorm' + str(idx)].dbeta

        for idx in range(1, self.affine_hidden_layer_num+2):
            grads['W' + str(idx + self.gc_hidden_layer_num)] = self.layers['Affine' + str(idx)].dW + \
                self.weight_decay_lambda_affine * self.params['W' +
                                                              str(idx + self.gc_hidden_layer_num)]
            grads['b' + str(idx + self.gc_hidden_layer_num)
                  ] = self.layers['Affine' + str(idx)].db

            if self.use_batchnorm and idx != self.affine_hidden_layer_num + 1:
                grads['Affine_gamma' +
                      str(idx)] = self.layers['Affine_BatchNorm' + str(idx)].dgamma
                grads['Affine_beta' +
                      str(idx)] = self.layers['Affine_BatchNorm' + str(idx)].dbeta

        return grads

    def set_params(self, params_dict):
        for key in params_dict.keys():
            if self.params[key].shape != params_dict[key].shape:
                print('ERROR!')
                return None
        for key in self.params.keys():
            self.params[key][...] = params_dict[key]
