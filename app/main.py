from flask import Flask, render_template, request, jsonify
from module.predict import predict
from module.make_feature import feature
import pickle
import os
app = Flask(__name__)
port = int(os.environ['PORT'])

with open("/app/models/ridgeCV_descriptors_dic_rdkit.sav", "rb") as f:
    descript_dic_origin = pickle.load(f)

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


@app.route("/model", methods=["POST"])
def model(): 
    descript_dic = descript_dic_origin.copy()
    molfile_dic = request.form
    
    # フラグメントの作成
    feature_dic, weight = feature(molfile_dic)
    
    # rdkitでmolファイルが読み込まれなかった時の処理
    if feature_dic is None:
        response = {'HOMO': None, 'LUMO': None}
        return jsonify(response)
    
    # 予測
    pre = predict(feature_dic, descript_dic, weight)
    response =  pre
    return jsonify(response)
 


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port)