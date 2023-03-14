import streamlit as st
import xgboost as xgb
import joblib
import pickle

def train_and_save_model(X_train, y_train, model_path):
    # Train XGBoost model
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    # Save model to file path
    # model.save_model(model_path)

    file = open("/Users/rohanmehra/code/roski10/Project_Mortgages/xgbmodel.pkl", "wb")
    pickle.dump(model, file)
    file.close()

    # Return model object
    return model




def load_and_predict(model_path, X_test):
    # Load model from file path
    # model = xgb.Booster()
    # model.load_model(model_path)
    model = joblib.load("xgbmodel.pkl")

    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Return predicted values
    return y_pred
