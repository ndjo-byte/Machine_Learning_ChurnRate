# evaluation.py created in ./Churn_Classification_Project/src



#imports 
import pickle
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import pandas as pd 

#import test data
df_test = pd.read_csv('../../data/test/customer_churn_dataset-testing-clean.csv')

#setting test X, y
X_test = df_test[['age', 'gender', 'tenure', 'usage_frequency', 'support_calls',
       'payment_delay', 'subscription_type', 'contract_length', 'total_spend',
       'last_interaction']]
y_test = df_test['churn']

#import model
with open('../models/FINAL_trained_model_02_DecisionTreeClass.pkl.pkl', 'rb') as f:
        model = pickle.load(f)

#classification report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_visual = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
cm_visual.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()


## churn precision recall
probability = model.predict_proba(X_test)[:, 1]  # churn probabilities
fpr, tpr, roc_thresholds = roc_curve(y_test, probability)
precision, recall, pr_thresholds = precision_recall_curve(y_test, probability)

# ROC curve
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()

# Precision-Recall curve
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 2)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve: Churn (1)')
plt.tight_layout() 


#feature importance 
importances = model.feature_importances_

feature_names = X_test.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Decision Tree Clasifier Feature Importance')
plt.gca().invert_yaxis()
plt.show()
