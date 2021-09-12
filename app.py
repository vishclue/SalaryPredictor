import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('salaryPredictor.mdl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    yExper = float(request.form['yExp'])
    yExpNpArray = np.array([[yExper]])
    sal = model.predict(yExpNpArray)
    #print("Salary for {} years of experience is $ {}".format(yExper, round(sal[0][0])))
    #print("Salary for {} years of experience is $ {}".format(yExper, round(sal[0][0])))

    

    return render_template('index.html', prediction_text="Salary for {} years of experience is $ {}".format(yExper, round(sal[0][0])))


if __name__ == "__main__":
    app.run(debug=True)