import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
import joblib


import datetime

from utils import *

df = pd.read_csv("dataset.csv")

df = clean_dataset(df)


# Add datetime field
df['date'] = df['MONAT'].apply(lambda x: x[:4] + "/" + x[4:])
df['date'] = pd.to_datetime(df['date'])


# Get only accidents due to alcohol.
af = df[df['MONATSZAHL'] == 'Alkoholunfälle']
af = af.iloc[:, 2:]

# Make series stationary

af = get_diff(af)

# Prepare the dataset to be ingested by the model
# Where every sample has features representing the deltas of the 12 previous months accidents
model_df = generate_supervised(af)


# Split the dataset into train and testing for evaluation. Year 2020 will be used for evaluation.
train, test = train_test_split(model_df)


# Scale the dataset using a minmax scaler
X_train, y_train, X_test, y_test, scaler_object = scale_data(train, test)


# Build and train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions for evaluation set
predictions = model.predict(X_test)

# Undo scaling for the test set
unscaled = undo_scaling(predictions, X_test, scaler_object)

# Transform the predictions from differences to actual accident numbers
unscaled_df = predict_df(unscaled, af)


# Print evaluation metrics
print(get_scores(unscaled_df, af))

# Plot predictions over real values
# plot_results(unscaled_df, af)


# Save model, scaler, and processed dataset for querying
joblib.dump(model, "model_files/model.pkl")
joblib.dump(scaler_object, "model_files/scaler.pkl")
af.to_csv("model_files/processed_alcohol_accidents.csv")