from flask import Flask, render_template, request, redirect
from details import models,categorical_columns
import numpy as np
import pickle
from PIL import Image
from keras.models import load_model

app = Flask(__name__,template_folder='templates')

@app.route('/')
def hello_world():
    return render_template('index.html',models=models)

@app.route('/predictor/brain',methods=['GET','POST'])
def bt_predictor():
    if request.method=='POST':

        loaded_model = load_model('./prediction_models/brain_tumor.h5')
        f = request.files['file']
        f.save("./static/images/important.jpeg") 
        img = Image.open("./static/images/important.jpeg")
        x = np.array(img.resize((128,128)))
        x = x.reshape(1,128,128,3)
        res = loaded_model.predict_on_batch(x)
        prediction = np.where(res == np.amax(res))[1][0]
        return render_template('form.html',name="brain",models=models,prediction=prediction,categorical_columns=categorical_columns)
    
    return render_template('form.html',name="brain",models=models,prediction=-1,categorical_columns=categorical_columns)

@app.route('/predictor/<name>',methods=['GET','POST'])
def predictors(name):
    if request.method=='POST':
        arr = []
        for attribute in models[name].attributes:
            arr.append((float)(request.form[attribute]))

        loaded_model = pickle.load(open(f"./prediction_models/{name}_model.sav",'rb'))
        input_data = np.asarray(arr)
        input_data_reshaped = input_data.reshape(1,-1)
        prediction = loaded_model.predict(input_data_reshaped)
        return render_template('form.html',name=name,models=models,prediction=prediction[0],categorical_columns=categorical_columns)

    return render_template('form.html',name=name,models=models,prediction=-1,categorical_columns=categorical_columns)

@app.route('/<name>')
def nav_links(name):
    return render_template(f"{name}.html")

if __name__ == '__main__':
    app.run(debug=True,port=8000)
