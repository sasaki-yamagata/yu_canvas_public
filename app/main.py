from flask import Flask, render_template, request, jsonify
import pickle
import os
from module.ridge_orb import RidgeOrb
from module.gcn_orb import GcnOrb

app = Flask(__name__)
port = int(os.environ['PORT'])
with open("/app/models/ridge/ridgeCV_descriptors_dic_rdkit.sav", "rb") as f:
    ml_feature_origin = pickle.load(f)

@app.route('/')
def index():
    return render_template(
      'index.html'
    ) 
@app.route('/en')
def index_e():
    return render_template(
      'index-e.html'
    )


@app.route("/predict", methods=["POST"])
def predict(): 
    ml_feature = ml_feature_origin.copy()
    molfile_dic = request.form
    
    #ridge回帰の予測値
    ridge_orb = RidgeOrb(molfile_dic)
    js_feature = ridge_orb.makeFeature()
    ridge_predict_orb = ridge_orb.predict(js_feature, ml_feature)
    print(ridge_predict_orb)
    
    # GCNの予測値
    gcn_orb = GcnOrb(molfile_dic)
    gcn_predict_orb = gcn_orb.predict()
    print(gcn_predict_orb)
    
    response = {'ridge_orb' : ridge_predict_orb, 'gcn_orb' : gcn_predict_orb}
    print(response)
    # for molfile in molfile_dic.values():
    #     if Chem.MolFromMolBlock(molfile) is None:
    #         return None, None
    #     else:
    #         mol = Chem.MolFromMolBlock(molfile)
    #         smiles = Chem.MolToSmiles(mol)
    # homo, lumo = GCN_predict(smiles)
    # print(f'homo: {homo}, lumo: {lumo}')
    
    # # フラグメントの作成
    # feature_dic, weight = feature(molfile_dic)
    
    # # rdkitでmolファイルが読み込まれなかった時の処理
    # if feature_dic is None:
    #     response = {'HOMO': None, 'LUMO': None}
    #     return jsonify(response)
    
    # # 予測
    # pre = predict(feature_dic, ml_feature, weight)
    # response =  pre
    return jsonify(response)
 


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port)