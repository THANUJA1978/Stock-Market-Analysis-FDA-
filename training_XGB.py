import os
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
import joblib
import time
import warnings

# Suppress UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UserWarning)


def train_model(file_path):
    # Load the data from the CSV file
    data = pd.read_csv(file_path)

    # Select features (X) and target variable (y)
    features = ['High', 'Low', 'Open', 'Volume', 'YTD Gains']
    X = data[features]
    y = data['Close']

    # Handle missing values in the target variable
    y = y.fillna(0)  # You can use a different strategy based on your data

    # Check if the model file already exists
    model_filename = os.path.join('stocks_model_XGB', f'{os.path.splitext(os.path.basename(file_path))[0]}.pkl')

    if os.path.exists(model_filename):
        print(f"Model already exists for {os.path.splitext(os.path.basename(file_path))[0]}. Skipping training.")
        return

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with an imputer and an XGBoost regressor
    model = make_pipeline(SimpleImputer(strategy='mean'), XGBRegressor(random_state=42))

    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()

    # Save the trained model
    joblib.dump(model, model_filename)

    # Print evaluation metrics
    y_pred = model.predict(X_test)
    
    # Evaluation metrics for regression
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Metrics for {os.path.splitext(os.path.basename(file_path))[0]}:")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    print(f"Training Time: {end_time - start_time} seconds")

    print(f"Model trained and saved for {os.path.splitext(os.path.basename(file_path))[0]}")

stocks_folder = "C:/Users/prama/OneDrive/Documents/Project/FDA_project/stocks"

for file_name in os.listdir(stocks_folder):
    if file_name.endswith('.csv'):
        file_path = os.path.join(stocks_folder, file_name)
        train_model(file_path)
