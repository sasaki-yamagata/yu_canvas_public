from flask import Flask, render_template, request, jsonify
from flask_box.predict import predict
import pickle
import os
app = Flask(__name__)
port = int(os.environ['PORT'])
descript_dic_origin = pickle.load(open("/app/models/ridgeCV_dic.sav", "rb"))

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
def plusone(): 
    descript_dic = descript_dic_origin.copy()
    feature_dic = request.form
    pre_orb = predict(feature_dic, descript_dic)
    response = pre_orb
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port)