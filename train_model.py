# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
insurance_dataset = pd.read_csv("insurance.csv")

# Encode categorical columns
insurance_dataset.replace({"sex":{"male":0, "female":1},
                           "smoker":{"yes":0, "no":1},
                           "region":{"southeast":0, "southwest":1, "northeast":2, "northwest":3}}, inplace=True)

X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

# Train model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Save model
pickle.dump(regressor, open('model.pkl', 'wb'))
