from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

app = Flask(__name__)

def root_dir():  
    return os.path.abspath(os.path.dirname(__file__))
root = root_dir() + '/'

le = LabelEncoder()
le = joblib.load(root+'le.pkl')
classifier = LGBMClassifier(objective = 'multiclass', max_depth = 6, n_estimators=100, random_state=42)
classifier = joblib.load(root+'classifier.pkl')
feature = ['gender', 'age']

@app.route('/ready')
def ready():
    return "OK"

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    data = pd.DataFrame({'gender': data['gender'], 'age': data['age'] }, index = [0])
    data['gender']=data['gender'].map({'U':2, 'F':1, 'M':0}).astype('int8')
    data['age'] = data['age'].fillna(0).clip(14, 90).astype('int8')

    try:
        y_pred = classifier.predict_proba(data[feature])
        best_n = np.argsort(y_pred, axis=1)[:,-30:]
        recommended = list(le.inverse_transform(best_n[0]))
    except:
        recommended = [
        '4009f09b04', '15ccaa8685', 'bf07df54e1', '3e038662c0', '4dcf79043e',
        'f4599ca21a', '5cb93c9bc5', '4a29330c8d', '439498bce2', '343e841aaa',
        '0a46068efc', 'dc2001d036', '31dcf71bbd', '5645789fdf', '113e3ace79',
        'f098ee2a85', '53fc95e177', '080ace8748', '4c07cb5835', 'ea27d5dc75',
        'cbe1cd3bb3', '1c257c1a1b', 'f5e18af323', '5186e12ff4', '6d0f84a0ac',
        'f95785964a', 'ad865591c6', 'ac81544ebc', 'de25bccdaf', 'f43c12d228',]

    return jsonify({'recommended_products': recommended })

if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True, port=8000)
