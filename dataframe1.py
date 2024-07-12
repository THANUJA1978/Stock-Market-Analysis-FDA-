import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def train_linear_regression(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def train_decision_tree(X_train, y_train, X_test):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def train_random_forest(X_train, y_train, X_test):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def train_xgboost(X_train, y_train, X_test):
    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def calculate_regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

def create_model_dataframe(models, X_train, y_train, X_test, y_test):
    model_data = {'Model': [], 'MAE': [], 'MSE': [], 'R2': []}
    
    for model_name, train_function in models.items():
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        y_pred = train_function(X_train_imputed, y_train, X_test_imputed)
        mae, mse, r2 = calculate_regression_metrics(y_test, y_pred)
        
        model_data['Model'].append(model_name)
        model_data['MAE'].append(mae)
        model_data['MSE'].append(mse)
        model_data['R2'].append(r2)
    
    df = pd.DataFrame(model_data)
    return df

def plot_regression_metrics(df):
    plt.figure(figsize=(10, 6))
    
    for index, row in df.iterrows():
        plt.plot(row.index[1:], row.values[1:], marker='o', label=row['Model'])

    plt.title('Regression Metrics Comparison')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.show()

# Load the data from the CSV file
data_path = "C:/Users/prama/OneDrive/Documents/Project/FDA_project/stocks/AAPL.csv"  # Replace with the actual path to your CSV file
data = pd.read_csv(data_path)

# Select features (X) and target variable (y)
features = ['High', 'Low', 'Open', 'Volume', 'YTD Gains']
X = data[features]
y = data['Close']

# Handle missing values in the target variable
y = y.fillna(0)  # You can use a different strategy based on your data

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define models
models = {
    'Linear Regression': train_linear_regression,
    'Decision Tree': train_decision_tree,
    'Random Forest': train_random_forest,
    'XGBoost': train_xgboost
}

# Train models and create a dataframe for metrics
model_df = create_model_dataframe(models, X_train, y_train, X_test, y_test)

# Display the dataframe
print(model_df)

# Plot regression metrics comparison
plot_regression_metrics(model_df)

