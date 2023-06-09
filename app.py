from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np
app = Flask(__name__)

lr = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods = ['POST'])
def predict():
    if request.method == "POST":

         Mean_Integrated = float(request.form['ab'])
         SD = float(request.form['sd'])
         EK =float(request.form['cd'])
         Skewness = float(request.form['ad'])
         Mean_DMSNR_Curve = float(request.form['ed'])
         SD_DMSNR_Curve = float(request.form['fd'])
         EK_DMSNR_Curve =float(request.form['gd'])
         Skewness_DMSNR_Curve = float(request.form['hd'])
         prediction = lr.predict([[Mean_Integrated,SD,EK,Skewness,Mean_DMSNR_Curve,SD_DMSNR_Curve,EK_DMSNR_Curve,Skewness_DMSNR_Curve]])
         pred = prediction[0]
         out = "Error"
         if pred==[1]:out= "Its a Pulsar"
         else: out = "Not a pulsar"
        
         return render_template('index.html',results = out)



if __name__ == '__main__':
    app.run(debug=True)