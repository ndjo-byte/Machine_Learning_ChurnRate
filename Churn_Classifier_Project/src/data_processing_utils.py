# data_processing_utils.py 

#imports
import pandas as pd 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# importing data
train_route = '../data/raw/customer_churn_dataset-training-master.csv'
test_route = '../data/raw/customer_churn_dataset-testing-master.csv'


def to_csv(df_route):
    
    df = pd.read_csv(df_route)


    return df


# cleaning data

def clean_df(df):
    # drop nulls
    df.dropna(axis=0, inplace=True)
    # dtypes to int
    df = df.astype({'CustomerID':'int', 'Age': 'int', 'Tenure':'int', 'Usage Frequency':'int', 'Support Calls':'int', 'Total Spend':'int', 'Last Interaction':'int',  'Churn':'int'})
    # remove index (ID col)
    df.drop(columns='CustomerID', inplace=True)
    #columns to numeric
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female':0});
    df['Subscription Type'] = df['Subscription Type'].map({'Basic':1, 'Standard':2,'Premium': 3})
    df['Contract Length'] = df['Contract Length'].map({'Monthly':1, 'Quarterly':3,'Annual':12 })
    #reorder cols 
    columns = [col for col in df.columns if col != 'Churn'] + ['Churn']
    df = df[columns]
    #normalise column names 
    df.rename(columns={'Age': 'age' , 'Gender': 'gender', 'Tenure': 'tenure', 'Usage Frequency': 'usage_frequency', 
                    'Support Calls':'support_calls', 'Payment Delay': 'payment_delay', 'Contract Length': 'contract_length', 
                    'Total Spend':'total_spend', 'Last Interaction': 'last_interaction', 'Subscription Type': 'subscription_type',
                    'Churn':'churn'}, inplace=True)
    #shuffle to ensure evenly distributed target value (churn)
    df = shuffle(df, random_state=42).reset_index(drop=True)

    return df 


def variable_distributor(df_train, df_test):
    combined_df = pd.concat([df_test, df_train], axis=0)
    combined_df = shuffle(combined_df, random_state=42)

    X = combined_df.drop(columns='churn')
    y = combined_df['churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    train_df = pd.concat([X_train, y_train], axis=1)

    test_df = pd.concat([X_test, y_test], axis=1)

    return train_df, test_df


