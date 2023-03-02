import pandas as pd
def clean_data(data: pd.DataFrame) -> pd.DataFrame:

    # We are only concerned with analysing the primary market for our project, therefore we must remove all rows associated with the action loan purchased by the instituiton.
    # We also need to remove all rows associated with the action 'Application withdrawn by client'



    data = data[~data['action_taken_name'].isin(['Application withdrawn by applicant', 'Loan purchased by the institution', 'File closed for incompleteness'])]
    data = data[~data['applicant_ethnicity_name'].isin(['Information not provided by applicant in mail, Internet, or telephone application'])]
    data = data[~data['applicant_race_name_1'].isin(['Information not provided by applicant in mail, Internet, or telephone application'])]
    data = data[~data['applicant_sex_name'].isin(['Information not provided by applicant in mail, Internet, or telephone application'])]
    data = data[~data['co_applicant_ethnicity_name'].isin(['Information not provided by applicant in mail, Internet, or telephone application'])]
    data = data[~data['co_applicant_race_name_1'].isin(['Information not provided by applicant in mail, Internet, or telephone application'])]
    data = data[~data['co_applicant_sex_name'].isin(['Information not provided by applicant in mail, Internet, or telephone application'])]

    data = data.dropna(subset=['county_name'])
    # new feature highlighting loan approved for 'loan originated' and not approved for everything else
    data['loan_status']=["approved" if x=="Loan originated" else "not approved" for x in data['action_taken_name']]

    # Drop irrelevant columns
    data = data.drop(columns=['applicant_race_name_5',
                          'applicant_race_name_4',
                          'applicant_race_name_3','applicant_race_name_2',
                          'co_applicant_race_name_5',
                          'co_applicant_race_name_4','co_applicant_race_name_3',
                          'co_applicant_race_name_2',
                          'denial_reason_name_3','denial_reason_name_2',
                          'denial_reason_name_1','rate_spread','edit_status_name',
                         'state_abbr','respondent_id','agency_abbr','as_of_year',
                            'application_date_indicator','state_name','sequence_number',
                         'census_tract_number'])
    return data
