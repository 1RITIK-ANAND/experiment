from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    file = request.files['file']
    df = pd.read_csv(file)
    predictions = model.predict(df)
    return render_template('index.html', prediction_text=f'Batch Prediction Completed. Predicted values: {predictions[:5]}...')  # show first 5



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = np.array(data).reshape(1, -1)
    prediction = model.predict(final_input)
    return render_template('index.html', prediction_text=f'Estimated Insurance Cost: ${prediction[0]:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
