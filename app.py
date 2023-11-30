from flask import Flask, request, render_template, send_file, make_response
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model1.pkl')  # Load your trained sklearn model

# Define the list of feature names
features = ['temp_laundry(T3)', 'temp_bath(T5)', 'temp_kitchen(T1)', 'temp_parents(T9)', 'temp_office(T5)',
            'temp_living(T2)', 'temp_teen(T8)', 'temp_iron(T8)', 'humid_kitchen(RH_1)', 'humid_office(RH_4)',
            'humid_bath(RH_5)', 'humid_living(RH_2)', 'humid_parents(RH_9)', 'humid_laundry(RH_3)', 'humid_teen(RH_8)',
            'humid_iron(RH_7)', 'temp_outside(T6)', 'humid_outside(RH_6)', 'temp_station(T_out)', 'humid_station(RH_out)',
            'session', 'Windspeed', 'Press_mm_hg']

@app.route('/')
def index():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    input_features = []
    for feature in features:
        value = request.form.get(feature)
        if value:
            input_features.append(float(value))
        else:
            input_features.append(0.0)

    input_features = np.array(input_features).reshape(1, -1)
    prediction = model.predict(input_features)
    fname='testcases.txt'
    with open(fname,'a') as file:
        for inp in input_features:
            print(inp, end= ',',file=file)
        print(prediction,file=file)  

    return render_template('index.html', features=features, prediction=f'Predicted Energy Consumption : {prediction[0]}')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
