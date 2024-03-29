# # --- Data Processing ---
# import pandas as pd
# from sklearn.pipeline import make_pipeline
# from sklearn.compose import make_column_transformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
# from sklearn.impute import SimpleImputer
# from imblearn.over_sampling import SMOTE
# import pickle

# def preprocess_and_resample(data: pd.DataFrame):

#     # Define the preprocessor
#     num_pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
#     num_col = ['tract_to_msamd_income','population','minority_population','number_of_owner_occupied_units',
#                'number_of_1_to_4_family_units','loan_amount_000s','hud_median_family_income','applicant_income_000s']

#     cat_pipe = OneHotEncoder()
#     cat_col = ['property_type_name','preapproval_name','owner_occupancy_name',
#                'loan_type_name','loan_purpose_name','lien_status_name',
#                'hoepa_status_name','co_applicant_sex_name','co_applicant_race_name_1',
#                'co_applicant_ethnicity_name','applicant_sex_name','applicant_race_name_1',
#                'applicant_ethnicity_name','agency_name', 'region']

#     targ_pipe = OrdinalEncoder(categories=[['not approved', 'approved']])
#     targ_col = ['loan_status']

#     preprocessor = make_column_transformer(
#         (num_pipe, num_col),
#         (cat_pipe, cat_col),
#         (targ_pipe, targ_col),
#         sparse_threshold=0.1,
#         remainder='passthrough'
#     )

#     # Apply the preprocessor
#     file = open("preproc.pkl", "wb")
#     pickle.dump(preprocessor,file)
#     file.close()

#     processed_data = pd.DataFrame(pickle.fit_transform(data), columns=preprocessor.get_feature_names_out())
#     X_processed = processed_data.drop(columns='ordinalencoder__loan_status')

#     print("\n✅ X_processed, with shape", X_processed.shape)

#     return X_processed
