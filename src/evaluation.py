import pandas as pd
from xgboost import XGBRegressor

from src.pipeline import build_pipeline

# Load the training and test data
df_train = pd.read_csv("./data/CW1_train.csv")
df_test = pd.read_csv("./data/CW1_test.csv")

# Separate the features and target variable
X_train = df_train.drop(columns=["outcome"])
Y_train = df_train["outcome"]

# Set the hyperparameters for the model
hyperparameters = {
    "subsample": 0.7,
    "reg_lambda": 1,
    "reg_alpha": 0.5,
    "n_estimators": 800,
    "min_child_weight": 10,
    "max_depth": 4,
    "learning_rate": 0.01,
    "gamma": 0,
    "colsample_bytree": 0.8
}

# Build the pipeline
model = build_pipeline(
    XGBRegressor(random_state=42, verbosity=0, n_jobs=-1, eval_metric="rmse", **hyperparameters)
)
model.fit(X_train, Y_train)

# Make predictions on the test data
Y_predictions = model.predict(df_test)
# Save the predictions to a CSV file
output = pd.DataFrame({"yhat": Y_predictions})
output.to_csv("CW1_submission_K23169225.csv", index=False)