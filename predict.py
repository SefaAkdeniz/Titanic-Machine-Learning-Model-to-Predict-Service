# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 19:09:25 2020

@author: sefa
"""

import numpy as np
import pandas as pd
import pickle
from flask import Flask
from flask import jsonify

app = Flask(__name__)

@app.route('/<int:Pclass>/<int:Sex>/<int:Age>/<int:SibSp>/<int:Parch>/<float:Fare>/<string:Embarked>')
def index(Pclass,Sex,Age,SibSp,Parch,Fare,Embarked):
    
    try:     
        if Embarked == "Q":
            array=pd.DataFrame([[Pclass,Sex,Age,SibSp,Parch,Fare,0,1,0]])
        elif Embarked == "C":
            array=pd.DataFrame([[Pclass,Sex,Age,SibSp,Parch,Fare,1,0,0]])
        elif Embarked == "S":
            array=pd.DataFrame([[Pclass,Sex,Age,SibSp,Parch,Fare,0,0,1]])
            
        scaler = pickle.load(open("scaler.sav", 'rb'))
        array = scaler.transform(array)
        
        model = pickle.load(open("finalized_model.sav", 'rb'))
        print(type(model))
        print(type(model.predict(array)[0]))
        
        return jsonify(str(model.predict(array)[0]))
    except:
        return jsonify(str("Hata"))
           
if __name__ == '__main__':
    app.run(debug=True)