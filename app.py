from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

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
