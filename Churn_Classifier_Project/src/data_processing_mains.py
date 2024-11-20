# data_processing.py created in ./Churn_Classification_Project/src

#imports
from src.data_processing_utils import to_csv, clean_df, variable_distributor



# importing data
train_route = '../data/raw/customer_churn_dataset-training-master.csv'
test_route = '../data/raw/customer_churn_dataset-testing-master.csv'


train_df = to_csv(train_route)
test_df = to_csv(test_route)


# cleaning training data

clean_train_df = clean_df(train_df)
clean_test_df = clean_df(test_df)

final_train_df, final_test_df = variable_distributor(clean_train_df, clean_test_df)


# saving
final_train_df.to_csv('../../data/train/customer_churn_dataset-training-clean.csv', index=False, header=True)
final_test_df.to_csv('../../data/test/customer_churn_dataset-testing-clean.csv', index=False, header=True) 