# training.py created in ./Churn_Classification_Project/src

#imports 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import pickle
import yaml


#import data
df_train = pd.read_csv('../../data/train/customer_churn_dataset-training-clean.csv')
df_test = pd.read_csv('../../data/test/customer_churn_dataset-testing-clean.csv')


# splitting train/test predictor X, y
X_train = df_train[['age', 'gender', 'tenure', 'usage_frequency', 'support_calls',
       'payment_delay', 'subscription_type', 'contract_length', 'total_spend',
       'last_interaction']]
y_train = df_train['churn']

X_test = df_test[['age', 'gender', 'tenure', 'usage_frequency', 'support_calls',
       'payment_delay', 'subscription_type', 'contract_length', 'total_spend',
       'last_interaction']]
y_test = df_test['churn']


#fitting DecisionTreeClassifier to training data

pipe = Pipeline(steps=[
    ('scaler', None),
    ('estimator', DecisionTreeClassifier(class_weight='balanced')),  
])

params = {
    'estimator__max_depth': [5, 10],
    'estimator__min_samples_split': [2, 5],
    'estimator__min_samples_leaf': [2, 5],
    'estimator__max_features': ['sqrt', None]
}

dtC_rs = RandomizedSearchCV(pipe, params, scoring='roc_auc', n_iter=10, cv=3, n_jobs=-1, random_state=42, verbose=3)
dtC_rs.fit(X_train, y_train)


#accessing optimised model
best_dtc = dtC_rs.best_estimator_


#saving model
with open('../../models/FINAL_trained_model_02_DecisionTreeClass.pkl', 'wb') as f:
    pickle.dump(best_dtc, f)

#saving model parameters
params = best_dtc.get_params()
with open('../../models/FINAL_trained_model_02_DecisionTreeClass.yml', 'w') as yaml_file:
    yaml.dump(params, yaml_file, default_flow_style=False)