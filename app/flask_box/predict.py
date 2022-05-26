import pickle
import numpy as np

def predict(feature_dic, descript_dic):
    '''
    feature_dic = keyがYU_canvasから入力された分子構造の記述子、valueが記述子の個数のdic
    descript_dic = keyが機械学習モデルの記述子、valueが個数がすべて0のdic
    return = homoとlumoの予測値をdicで返している
    '''
    
    # 重みでモデルを変更
    weight = float(feature_dic['weight'])
    if weight <= 287:
        weight_div = 0
    elif weight <= 369:
        weight_div = 1
    elif weight <= 486:
        weight_div = 2
    else:
        weight_div = 3
        
    # モデルを読み込む
    with open(f"/app/models/4div_{weight_div}_ridgeCV_fragment_depth2_all_cleaning_HOMO.sav", "rb") as f:
        model_homo = pickle.load(f)
    with open(f"/app/models/4div_{weight_div}_ridgeCV_fragment_depth2_all_cleaning_LUMO.sav", "rb") as f:    
        model_lumo = pickle.load(f)
      
    for feature in feature_dic:
        
    # 分子量の場合はスキップ
        if feature == 'weight':
            continue
        
    # descript_dicにfeature_dicの個数を追加
        if feature in descript_dic:
            descript_dic[feature] = feature_dic[feature]
            
    # 個数のみの配列を作成し、モデルに挿入      
    descript_count = list(descript_dic.values())
    descript_array = np.array([descript_count], dtype=int)

    # 予測値をdicに格納
    homo = model_homo.predict(descript_array)[0]
    lumo = model_lumo.predict(descript_array)[0]
    pre_orb = {'homo': homo, 'lumo': lumo}
    return pre_orb


if __name__ == "__main__": 
  predict()