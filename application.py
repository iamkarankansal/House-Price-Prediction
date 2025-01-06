import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn

application = Flask(__name__)
app = application

## Load the model
model = pickle.load(open('model/model.pkl', 'rb'))

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Get form data
        MedInc = float(request.form['MedInc'])
        HouseAge = float(request.form['HouseAge'])
        AveRooms = float(request.form['AveRooms'])
        Population = float(request.form['Population'])
        AveOccup = float(request.form['AveOccup'])
        Latitude = float(request.form['Latitude'])
        Longitude = float(request.form['Longitude'])

        # Create a DataFrame for the input data
        input_data = pd.DataFrame([[MedInc, HouseAge, AveRooms, Population, AveOccup, Latitude, Longitude]],
                                  columns=['MedInc', 'HouseAge', 'AveRooms', 'Population', 'AveOccup', 'Latitude', 'Longitude'])

        # Standardize the input data
        scaler = StandardScaler()
        input_data_scaled = scaler.fit_transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)
        price = prediction[0]

        return render_template('home.html', price=price)
    else:
        return render_template('home.html')

@app.route("/")
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)