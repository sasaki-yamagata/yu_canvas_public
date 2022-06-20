import pickle
import numpy as np

def predict(feature_dic, descript_dic, weight):
    '''
    feature_dic = keyがYU_canvasから入力された分子構造の記述子、valueが記述子の個数のdic
    descript_dic = keyが機械学習モデルの記述子、valueが個数がすべて0のdic
    return = homoとlumoの予測値をdicで返している
    '''
    
    # 重みでモデルを変更
    if weight <= 287:
        weight_div = 0
    elif weight <= 369:
        weight_div = 1
    elif weight <= 486:
        weight_div = 2
    else:
        weight_div = 3
        
    # モデルを読み込む
    with open(f"/app/models/4div_{weight_div}_ridgeCV_fragment_depth2_all_rdkit_HOMO.sav", "rb") as f:
        model_homo = pickle.load(f)
    with open(f"/app/models/4div_{weight_div}_ridgeCV_fragment_depth2_all_rdkit_LUMO.sav", "rb") as f:    
        model_lumo = pickle.load(f)
    
    match_count = 0
    for feature in feature_dic:   
    # descript_dicにfeature_dicの個数を追加
        if feature in descript_dic:
            descript_dic[feature] = feature_dic[feature]
            match_count += 1
            print(f'matching: {feature}')
        else:
            print(f'No match: {feature}')
    rate = (match_count / len(feature_dic)) * 100
    print(rate)
    
            
    # 個数のみの配列を作成し、モデルに挿入
    # descript_dic1 = {**descript_dic, **feature_dic}
    # if len(descript_dic1) != len(descript_dic):
    #     pre_orb = {'homo': 0, 'lumo': 0}
    #     return pre_orb
    if 2 in descript_dic.values():
        print('入ってる')
    descript_count = list(descript_dic.values())
    descript_array = np.array([descript_count], dtype=int)

    # 予測値をdicに格納
    homo = model_homo.predict(descript_array)[0]
    lumo = model_lumo.predict(descript_array)[0]
    pre_orb = {'homo': homo, 'lumo': lumo, 'rate': rate}
    return pre_orb


if __name__ == "__main__":
  predict()