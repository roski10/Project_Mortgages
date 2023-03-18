# --- Data Processing ---
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pickle

def preprocess_and_resample(data: pd.DataFrame):
    # map 'approved' to 1 and 'not approved' to 0
    status_map = {'approved': 1, 'not approved': 0}
    data['loan_status'] = data['loan_status'].map(status_map)

    # Split the data
    X = data.drop(columns='loan_status')
    y = data['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Define the preprocessor
    num_pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
    num_col = ['tract_to_msamd_income','population','minority_population','number_of_owner_occupied_units',
               'number_of_1_to_4_family_units','loan_amount_000s','hud_median_family_income','applicant_income_000s']
    cat_pipe = OneHotEncoder()
    cat_col = ['property_type_name','preapproval_name','owner_occupancy_name',
               'loan_type_name','loan_purpose_name','lien_status_name',
               'hoepa_status_name','co_applicant_sex_name','co_applicant_race_name_1',
               'co_applicant_ethnicity_name','applicant_sex_name','applicant_race_name_1',
               'applicant_ethnicity_name', 'region']

    preprocessor = make_column_transformer(
        (num_pipe, num_col),
        (cat_pipe, cat_col),
        sparse_threshold=0.1,
        remainder='passthrough'
    )

    # Apply the preprocessor to the training data
    X_train_processed = pd.DataFrame(preprocessor.fit_transform(X_train), columns=[name.split('__')[1] for name in preprocessor.get_feature_names_out()])

    # Resample the training data with SMOTE
    sm = SMOTE(sampling_strategy='minority')
    X_train_sm, y_train_sm = sm.fit_resample(X_train_processed, y_train)

    # Apply the preprocessor to the test data
    X_test_processed = pd.DataFrame(preprocessor.transform(X_test), columns=preprocessor.get_feature_names_out())

    print(f"\n✅ X_train_sm, with shape {X_train_sm.shape}")
    print(f"✅ X_test_processed, with shape {X_test_processed.shape}")
    print(f"✅ y_train_sm, with shape {y_train_sm.shape}")
    print(f"✅ y_test, with shape {y_test.shape}")

    return X_train_sm, X_test_processed, y_train_sm, y_test
