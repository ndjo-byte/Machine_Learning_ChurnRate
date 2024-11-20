# Machine Learning 
## Classification Models

## Context: Churn Rate

**Churn**: churn defines when a client leaves a service or stops using a product.

**Impact on Key Performance Indicators (KPIs)**: income, retention, growth.

**Problem**: it is difficult to identify clients at risk of abandoning a service and damaging KPIs.

**Solucion**: _predict_ clients at risk of abandoning a service, by means of machine learning, thus enabling companies to implement target preventative, retention measures. 

## Models and Structure

- Making use of Sklearn and Keras, 7 models have been trained, all accessible in .pkl format in **#models**. Each has been optimised using RandomSearchCV or other methods. 
    - Logistic Regression
    - Decision Tree Classifier 
    - Random Forest Classifier 
    - KNN (unsupervised) 
    - SVC (with PCA to reduce workload)
    - Neural Network 

- The workflow and evaluation for each model is located in **#notebooks**; the best performing model (Decision Tree Classifier) is also trained and evaluated in two separate .py files located in #src folder.

- Likewise, data cleaning and engineering is contained in **#notebooks**; also available as .py scripts in **#src folder**. 

- The best performing model (Decision Tree) is deployable in **#app_streamlit**. 

- Raw and clean data is located in **#data**. The data was sourced from Kaggle. 

- Finally, technical and business presentations used to communicate my findings to stakeholders are located in **#docs**. 


