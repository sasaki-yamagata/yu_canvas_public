import pickle
import numpy as np

def predict(feature_dic, descript_dic):
    weight = float(feature_dic['weight'])
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
    with open(f"/app/models/4div_{weight_div}_ridgeCV_fragment_depth2_all_cleaning_HOMO.sav", "rb") as f:
        model_homo = pickle.load(f)
    with open(f"/app/models/4div_{weight_div}_ridgeCV_fragment_depth2_all_cleaning_LUMO.sav", "rb") as f:    
        model_lumo = pickle.load(f)

    for feature in feature_dic:
        if feature == 'weight':
            continue

        if feature in descript_dic:
            descript_dic[feature] = feature_dic[feature]
    descript_count = list(descript_dic.values())

    descript_array = np.array([descript_count], dtype=int)

    homo = model_homo.predict(descript_array)[0]
    lumo = model_lumo.predict(descript_array)[0]
    pre_orb = {'homo': homo, 'lumo': lumo}
    return pre_orb


if __name__ == "__main__": 
  predict()