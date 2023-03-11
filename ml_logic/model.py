import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from ml_logic.preprocessing import preprocess_and_resample

def train_and_evaluate_model(X_train: pd.DataFrame,
                             X_test: pd.DataFrame,
                             y_train,
                             y_test,
                             objective='binary:logistic',
                             learning_rate= 0.3,
                             n_estimators=200,
                             reg_alpha=0.1,
                             reg_lambda=0.5):
    """
    Trains a random forest classifier on the training data, using the specified hyperparameters,
    and evaluates its performance on the test data.
    """
    # Define the random forest classifier
    model = XGBClassifier(objective=objective, learning_rate=learning_rate, n_estimators=n_estimators, reg_alpha=reg_alpha, reg_lambda=reg_lambda)

    # Train the classifier on the training data
    model.fit(X_train, y_train)


    print(f"\n✅ Model trained on {len(X_train)} rows of training data.")

    if model is None:
        print(f"\n❌ no model to evaluate")
        return None

    # Use the trained model to predict the target values for the test data
    y_pred = model.predict(X_test)

    # Compute the accuracy score for the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy score: {accuracy:.4f}")

    # Print the classification report for the model
    class_report = classification_report(y_test, y_pred)
    print(f"\n✅ Classification report:\n{class_report}")

    return model, y_pred
