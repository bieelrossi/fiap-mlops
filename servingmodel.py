import joblib
import pandas as pd
import json
import numpy as np
from flask import Flask, jsonify, request
import sys

app = Flask(__name__)

modelo = None

@app.route("/predict", methods=['GET', 'POST'])
def call_predict(request = request):
    print(request.values)

    json_ = request.json
    campos = pd.DataFrame([json_])
    
    if campos.shape[0] == 0:
        return "Dados de chamada da API estão incorretos.", 400

    prediction = modelo.predict(campos)

    return jsonify(str([1 if prob > 0.5 else 0 for prob in prediction]))

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 1:
        args.append('models/inadimplencia_lgbm.joblib')
    if len(args) < 2:
        args.append('8080')

    print(args)

    modelo = joblib.load(args[0])
    # app.run(port=8080, host='0.0.0.0')
    app.run(port=args[1], host='0.0.0.0', debug=True)
    pass

