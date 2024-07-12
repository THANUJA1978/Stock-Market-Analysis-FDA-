import os
import pandas as pd
import joblib
import warnings

# Suppress UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UserWarning)

def predict_and_choose_best_investment(data_folder, model_folder):
    # Initialize variables to track the highest returns and corresponding company
    max_returns = float('-inf')
    best_company = None

    # Iterate through each trained model
    for model_filename in os.listdir(model_folder):
        if model_filename.endswith('.pkl'):
            model_path = os.path.join(model_folder, model_filename)

            # Load the trained model
            loaded_model = joblib.load(model_path)

            # Extract company name from the model filename
            company_name = os.path.splitext(model_filename)[0]

            # Load the new data for the current company
            new_data_path = os.path.join(data_folder, 'new_data', f'new_data_{company_name}.csv')
            new_data = pd.read_csv(new_data_path)

            # Assume 'Open', 'High', 'Low', 'Volume', 'Date', 'YTD Gain' are the features for prediction (exclude 'Close')
            X_new = new_data[['High', 'Low', 'Open', 'Volume', 'YTD Gains']]

            # Make predictions on the new data
            close_predictions = loaded_model.predict(X_new)

            # Calculate potential returns based on predicted Close values
            new_data['Predicted Close'] = close_predictions

            # Display the new_data DataFrame with predicted 'Close' values and returns for the current company
            print(f"\nPredicted Close and Returns for {company_name}:")
            print(new_data)

            # Calculate daily returns based on historical stock data in 'stocks' folder
            stocks_folder = os.path.join(data_folder, 'stocks')
            stock_data_path = os.path.join(stocks_folder, f'{company_name}.csv')
            historical_data = pd.read_csv(stock_data_path)

            # Calculate 'pct_change()' for historical Close values
            historical_data['Daily Returns'] = historical_data['Close'].pct_change()
            historical_data['Daily Returns'].fillna(0, inplace=True)

            # Calculate average daily returns
            avg_daily_returns = historical_data['Daily Returns'].mean()

            # Display average daily returns for the current company
            print(f"\nAverage Daily Returns for {company_name}: {avg_daily_returns}")

            # Compare and update the highest average daily returns and best company
            if avg_daily_returns > max_returns:
                max_returns = avg_daily_returns
                best_company = company_name

    # Display the best company with the highest average daily returns
    print(f"\nThe best company to invest in is: {best_company}")
    print(f"The average daily returns for {best_company} is: {max_returns}")

# Assuming your new data CSV files are named 'new_data_company1.csv', 'new_data_company2.csv', etc.
data_folder = "C:/Users/prama/OneDrive/Documents/Project/FDA_project"
# Assuming your trained models are in the 'stocks' folder
model_folder = "C:/Users/prama/OneDrive/Documents/Project/FDA_project/stocks_model_random"
predict_and_choose_best_investment(data_folder, model_folder)

