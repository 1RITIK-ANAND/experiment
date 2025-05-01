# -*- coding: utf-8 -*-
"""HEALTH_INSURANCE_COST_PREDICTION.ipynb
Original file is located at
    https://colab.research.google.com/drive/1LouEQEaEzZ2BdRNHNFL8OyrXYIIVY1oO
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression

"""Data Collection & Analysis"""

insurance_dataset = pd.read_csv('/content/insurance.csv')

insurance_dataset.head()

#insurance_dataset.shape()
#will give total no. of rows and columns
#insurance_datatest.info()

insurance_dataset.info()

"""Cateogrical Features:
- Sex
- Smoker
- Region
"""

# checking for missing values
insurance_dataset.isnull().sum()

"""Data Analysis"""

# statistical MMeasures Of the dataset
insurance_dataset.describe()

#distribution of age value
sns.set()
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['age'])
plt.title('Age Distribution')
plt.show()

#Gender Distribution
plt.figure(figsize=(6,6))
sns.countplot(data=insurance_dataset, x="sex")
plt.title('Sex Distribution')
plt.show()

insurance_dataset['sex'].value_counts()

#Bmi Distribution
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['bmi'])
plt.title('BMI Distribution')
plt.show()

# Setup figure
fig, axs = plt.subplots(3, 3, figsize=(18, 12))  # 3 rows, 3 cols
plt.suptitle("All Column-wise Visualization", fontsize=18)

# Plot 1: Age
sns.histplot(insurance_dataset['age'], kde=True, ax=axs[0, 0])
axs[0, 0].set_title("Age Distribution")

# Plot 2: Sex
sns.countplot(x='sex', data=insurance_dataset, ax=axs[0, 1])
axs[0, 1].set_title("Sex Count")

# Plot 3: BMI
sns.histplot(insurance_dataset['bmi'], kde=True, ax=axs[0, 2])
axs[0, 2].set_title("BMI Distribution")

# Plot 4: Children
sns.countplot(x='children', data=insurance_dataset, ax=axs[1, 0])
axs[1, 0].set_title("Children Count")

# Plot 5: Smoker
sns.countplot(x='smoker', data=insurance_dataset, ax=axs[1, 1])
axs[1, 1].set_title("Smoker Count")

# Plot 6: Region
sns.countplot(x='region', data=insurance_dataset, ax=axs[1, 2])
axs[1, 2].set_title("Region Count")

# Plot 7: Charges
sns.histplot(insurance_dataset['charges'], kde=True, ax=axs[2, 0])
axs[2, 0].set_title("Charges Distribution")

# Hide unused plots (2 remaining slots in 3x3 grid)
axs[2, 1].axis('off')
axs[2, 2].axis('off')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#children column
plt.figure(figsize=(7,7))
sns.countplot(x='children', data=insurance_dataset)
plt.title('Children Column')
plt.show()

insurance_dataset['children'].value_counts()

"""Encoding the categorical features"""

#encoding sex column
insurance_dataset.replace({"sex":{"male":0, "female":1}}, inplace=True)

#encoding "somker" column
insurance_dataset.replace({"smoker":{"yes":0, "no":1}}, inplace=True)

#encoding "region" column
insurance_dataset.replace({"region":{"southeast":0, "southwest":1, "northeast":2, "northwest":3}}, inplace=True)

insurance_dataset

"""Splitting the feature and target"""

X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

print(X)

print(Y)

"""Splitting the data into Training data & Testing data  """

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

"""Model Training

Linear Regression
"""

regressor = LinearRegression()

regressor.fit(X_train, Y_train)

"""Model Evaluation"""

#prediction on training data
training_data_prediction = regressor.predict(X_train)

r2_train= metrics.r2_score(Y_train, training_data_prediction)
print("R squared vale : ", r2_train)

#prediction on test data
test_data_prediction = regressor.predict(X_test)

# R squared value
r2_test= metrics.r2_score(Y_test, test_data_prediction)
print("R squared vale : ", r2_test)

"""Predictive System"""

input_data = (32,1,25.74,0,1,0)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = regressor.predict(input_data_reshaped)
print(prediction)