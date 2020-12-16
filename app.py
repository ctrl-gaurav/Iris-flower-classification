from flask import Flask, render_template , url_for , request , redirect
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# loading trained model
flower_model=load_model('final_iris_model.h5')
flower_scaler=joblib.load("iris_scaler.pkl")

# initializing flask application
app=Flask(__name__)

# home page route(index.html)
@app.route('/')
def home():
    return render_template('index.html')

# when submitting the form 
@app.route('/form_action', methods=['POST','GET'])
def formAction():
    
    if request.method =='POST':
        values = request.form
        content={}
        content['sepal_length'] = float(values.get("sLength")) 
        content['sepal_width'] = float(values.get("sWidth"))
        content['petal_length'] = float(values.get("pLength"))
        content['petal_width'] = float(values.get("pWidth"))

        results = return_prediction(flower_model,flower_scaler,content) #stores the predicted flower name in results var
        return render_template('index.html', result = results) #return to the index page with results placeholder changed to the predicted name

    return render_template("index.html", result = "An error occured try again")

# func to predit the flower from the values got from form and returning the predicted flower name
def return_prediction(model,scaler,sample_json):
    
    s_len=sample_json["sepal_length"]
    s_wid=sample_json["sepal_width"]
    p_len=sample_json["petal_length"]
    p_wid=sample_json["petal_width"]
    
    flower=[[s_len,s_wid,p_len,p_wid]]
    
    classes = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    flower=scaler.transform(flower)
    class_ind=model.predict_classes(flower)[0]
    
    return classes[class_ind] # returning the predicted flower name

# runs the app
if __name__ == '__main__':
    app.run()    