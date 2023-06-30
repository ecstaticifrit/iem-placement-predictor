from flask import Flask, render_template, request
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the pre-trained ML model
with open('iem_model','rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the user inputs from the form
    gender = int(request.form['dropdown1'])
    b_10 = int(request.form['dropdown2'])
    b_12 = float(request.form['dropdown3'])
    adm_mode = float(request.form['dropdown4'])
    stream = int(request.form['dropdown5'])

    p_10 = float(request.form['input1'])
    p_12 = float(request.form['input2'])
    back = float(request.form['input3'])
    gpa = float(request.form['input4'])


    # Prepare the input data for prediction
    input = np.array([stream, gender, p_10, b_10, p_12, b_12, back, adm_mode, gpa])


    # Perform prediction using the loaded model
    prediction = model.predict([input])
 
    # Interpret the prediction result
    if prediction == 0 and back > 0:
        result = 'You will be placed only if backlogs are cleared'
    elif prediction == 0 and back == 0:
        result = 'You will be placed'
    elif prediction == 1:
        result = 'You will not be placed'

    return result

if __name__ == '__main__':
    app.run()

#app.run(debug=True)
