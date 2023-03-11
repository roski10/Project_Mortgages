# --- Data manipulation ---
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize

def clean_data(data: pd.DataFrame):
    # Define list of columns to filter for "Information not provided" values
    filter_cols = ['applicant_ethnicity_name', 'applicant_race_name_1', 'applicant_sex_name',
                   'co_applicant_ethnicity_name', 'co_applicant_race_name_1', 'co_applicant_sex_name']

    # Filter for rows where "action_taken_name" is not equal to certain values
    data = data.loc[~data['action_taken_name'].isin(['Application withdrawn by applicant', 'Loan purchased by the institution', 'File closed for incompleteness'])]

    # Filter for rows where values in specified columns are not equal to "Information not provided"
    for col in filter_cols:
        data = data.loc[~data[col].isin(['Information not provided by applicant in mail, Internet, or telephone application'])]

        # Create a dictionary of county to region mappings
    county_to_region = {
        'Whatcom County': 'Northern Cascades','Skagit County': 'Northern Cascades','Snohomish County': 'Northern Cascades',
        'King County': 'Western Region','Pierce County': 'Western Region','Kitsap County': 'Western Region',
        'Island County': 'Western Region','San Juan County': 'Western Region','Jefferson County': 'Western Region',
        'Clallam County': 'Western Region','Mason County': 'Olympic Peninsula','Clark County': 'Southwest Washington',
        'Cowlitz County': 'Southwest Washington','Wahkiakum County': 'Southwest Washington','Skamania County': 'Southwest Washington',
        'Adams County': 'Eastern Washington','Asotin County': 'Eastern Washington','Benton County': 'Eastern Washington',
        'Chelan County': 'Eastern Washington','Columbia County': 'Eastern Washington','Douglas County': 'Eastern Washington',
        'Ferry County': 'Eastern Washington','Franklin County': 'Eastern Washington','Garfield County': 'Eastern Washington',
        'Grant County': 'Eastern Washington','Kittitas County': 'Eastern Washington','Klickitat County': 'Eastern Washington',
        'Lincoln County': 'Eastern Washington','Okanogan County': 'Eastern Washington','Pend Oreille County': 'Eastern Washington',
        'Spokane County': 'Eastern Washington','Stevens County': 'Eastern Washington','Walla Walla County': 'Eastern Washington',
        'Whitman County': 'Eastern Washington','Yakima County': 'Eastern Washington','Thurston County':'Western Region',
        'Lewis County': 'Western Region','Grays Harbor County': 'Western Region','Pacific County': 'Southwest Washington',
        'Seattle, Bellevue, Everett': 'Western Region','Tacoma, Lakewood':'Northern Cascades','Portland, Vancouver, Hillsboro':'Northern Cascades',
        'Spokane, Spokane Valley':'Eastern Washington'
    }

    # Add a new column to your dataframe containing the region for each county
    data['region'] = data['county_name'].map(county_to_region)

    # Drop rows with missing values in "county_name" column
    data = data.dropna(subset=['county_name'])

    # Create new column "loan_status" based on "action_taken_name"
    data['loan_status'] = np.where(data['action_taken_name'] == 'Loan originated', 'approved', 'not approved')

    # Drop irrelevant columns
    drop_cols = ['applicant_race_name_5', 'applicant_race_name_4', 'applicant_race_name_3',
                 'applicant_race_name_2', 'co_applicant_race_name_5', 'co_applicant_race_name_4',
                 'co_applicant_race_name_3', 'co_applicant_race_name_2', 'denial_reason_name_3',
                 'denial_reason_name_2', 'denial_reason_name_1', 'rate_spread', 'edit_status_name',
                 'state_abbr', 'respondent_id', 'agency_abbr', 'as_of_year', 'application_date_indicator',
                 'state_name', 'sequence_number', 'census_tract_number', 'action_taken_name', 'purchaser_type_name',
                 'county_name','msamd_name']

    data = data.drop(columns=drop_cols)

    # Winsorize numeric columns
    data.select_dtypes(exclude=['object']).apply(lambda x: winsorize(x, limits=[0.05, 0.05]), axis=0, raw=True)

    # create a list of columns to be converted
    cols_to_convert = [
        'tract_to_msamd_income','population','minority_population','number_of_owner_occupied_units',
        'number_of_1_to_4_family_units','loan_amount_000s','hud_median_family_income','applicant_income_000s'
    ]

    # use the astype() method to convert the dtype of columns
    data[cols_to_convert] = data[cols_to_convert].astype('float')

    print("\n✅ data cleaned")

    return data
