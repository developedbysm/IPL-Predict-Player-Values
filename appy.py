import numpy as np
import pickle
from flask import Flask,request,render_template

app=Flask(__name__, template_folder='template')
model=pickle.load(open('ipl_pickle1.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction=model.predict(final_features)
    
    output = round(prediction[0],2)
    
    return render_template('predict.html',prediction_value='${}'.format(output)) 
    
if __name__=="__main__":
    app.run(debug=True)
