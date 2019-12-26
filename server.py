from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from collections import Counter

app = Flask(__name__)

def root_dir():  
    return os.path.abspath(os.path.dirname(__file__))
root = root_dir() + '/'

le = LabelEncoder()
le = joblib.load(root+'le.pkl')
prod_enc = LabelEncoder()
prod_enc = joblib.load(root+'prod_enc.pkl') 
classifier = LGBMClassifier(objective = 'multiclass', num_leaves=4, n_estimators=100, random_state=42,
                           subsample_freq=1, subsample=0.75, colsample_bytree=0.9, learning_rate=0.05)
classifier = joblib.load(root+'classifier.pkl')
c = Counter()
feature = ['gender', 'age', 'popular_product', 'product_len']

@app.route('/ready')
def ready():
    return "OK"

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    #собираем из запроса необходимые данные
    products_count = 0
    transaction_count = 0
    for tr in data['transaction_history']:
      products_count +=1
      #purchase_sum += tr['purchase_sum']

      for product in tr['products']:
        transaction_count +=1
        #quantity += pr['quantity']
        c[product['product_id']] += 1
    
    #кодируем самый популярный у пользователя продукт для модели
    try:
       popular_product = prod_enc.transform([product for product, count in c.most_common(1)])
    except:    
       popular_product = prod_enc.transform(['4009f09b04'])
    #print (popular_product)

    if products_count > 0:
        product_len = transaction_count // products_count
    else: 
        product_len = 0

    #собираем данные в датафрейм того же формата, на котором тренировалась модель    
    data = pd.DataFrame({'gender': data['gender'], 'age': data['age'], 'popular_product': popular_product, 'product_len': product_len, }, index = [0])
    data['gender']=data['gender'].map({'U':2, 'F':1, 'M':0}).astype('int8')
    data['age'] = data['age'].fillna(0).clip(14, 90).astype('int8')

    
    #определяем длину рекомендаций как среднее значение длины списка покупок в истории пользователя
    #n = int(data['product_len'].clip(1, 30))
    n = 30

    try:
        y_pred = classifier.predict_proba(data[feature])
        best_n = np.argsort(-y_pred, axis=1)[:,0:n]
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
