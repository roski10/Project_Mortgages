import os
import pandas as pd
from clean_data import clean_data
from preprocessing import  preprocess_and_resample
from model import train_and_evaluate_model


def main(file_path):
    data_original = pd.read_csv(file_path, decimal=',')
    data = clean_data(data_original)
    X_train, X_test, y_train, y_test = preprocess_and_resample(data)
    model, y_pred = train_and_evaluate_model(X_train, X_test, y_train, y_test)

    return model, y_pred

if __name__ == '__main__':
    data_dir = "raw_data"
    file_name = "Washington_State_HDMA-2016.csv"
    file_path = os.path.join(data_dir, file_name)
    main(file_path)
