# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 20:13:23 2022

@author: Akshat Dhir
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import plot_confusion_matrix
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def cutoff_predict(classifier, input_data, cutoff):
    return (classifier.predict_proba(input_data)[:,1]>cutoff).astype(int)
    
def main():
    
# =============================================================================
#       Data load, no default {80368} : default {19632}
# =============================================================================
    
    df = pd.read_excel(".\PastLoans.xlsx") #base_df   
    result_df_stage1 = pd.read_excel(".\Result_Stage1_14.xlsx") #result_df
    result_stage1 = result_df_stage1['default']
    output_df_stage1 = pd.read_excel(".\LoanApplications_Stage1_Lender1.xlsx")
    output_df_stage1.drop('employment', inplace=True, axis=1)
    output_df_stage1['default'] = result_stage1
     
# =============================================================================
#       Data Preperation for social1
# =============================================================================
    
    df.drop('social2', inplace=True, axis=1)
    df.drop('social3', inplace=True, axis=1) #lazy at 2AM
    training_df = pd.concat([df, output_df_stage1])
    training_df.drop(df[df.income > 150000].index, inplace=True)
    # sc = MinMaxScaler()
    training_df = training_df[['id', 'sex', 'Employed', 'Other', 'marital', 'income', 'social1', 'default',
           ]]
    X2 = training_df.iloc[:, 1:7].values
    Y2 = training_df.iloc[:, 7].values
    # X2 = sc.fit_transform(X2,Y2)

    
# # =============================================================================
# #     Classifier + SMOTE using best results from above + social1 only
# # =============================================================================
    
    oversample = SMOTE()
    over_X2, over_Y2 = oversample.fit_resample(X2, Y2)
    over_X_train2, over_X_test2, over_Y_train2, over_Y_test2 = train_test_split(over_X2, over_Y2, test_size=0.3, stratify = over_Y2)
    classifier4 = BalancedRandomForestClassifier(max_depth=10, min_samples_leaf=50, n_estimators=200, n_jobs=-1, random_state=42, class_weight = 'balanced_subsample' )
    classifier4.fit(over_X_train2, over_Y_train2)
        
    # # =============================================================================
    # #     Results
    # # =============================================================================
    
    imp_df = pd.DataFrame({
    "Varname": training_df.columns.drop(['default', 'id']),
    "Importance": classifier4.feature_importances_
    })
    print(imp_df.sort_values(by=["Importance"], ascending=False))
    Y_pred3 = cutoff_predict(classifier4, over_X_test2, 0.5)
    Y_pred3_probabilities = classifier4.predict_proba(over_X_test2)
    fig4 = plot_confusion_matrix(classifier4, over_X_test2, over_Y_test2, display_labels=['Will Not Default', 'Will Default'], cmap='Greens')
    plt.title('Standard Random Forest Confusion Matrix')
    plt.show()
    print("Accuracy score for SMOTE social1 model is " + str(accuracy_score(over_Y_test2, Y_pred3)*100)+"%")
        
    print('ROCAUC score for SMOTE social1 model is:',roc_auc_score(over_Y_test2, Y_pred3))
    print('Report:',classification_report(over_Y_test2, Y_pred3))
    print('F1 score:',f1_score(over_Y_test2, Y_pred3))
        
        
    
# =============================================================================
#     Let's make the new excel
# =============================================================================

    output_df = pd.read_excel(".\LoanApplications_Stage2_Lender1.xlsx")
    output_df.drop('employment', inplace=True, axis=1)
    output_df = output_df[['id', 'sex', 'Employed', 'Other', 'marital', 'income', 'social1'
            ]]
    output_X = output_df.iloc[:, 1:7].values
    Y_pred4 = cutoff_predict(classifier4, output_X, 0.5)
    Y_pred4_probabilities = (classifier4.predict_proba(output_X))
    output_df['default'] = Y_pred4
    output_df['probability of no default'] = Y_pred4_probabilities[:,0]
    output_df['probability of default'] = Y_pred4_probabilities[:,1]
    output_df.to_excel(".\LoanApplications_Stage2_result1.xlsx", index = False)   
     
if __name__ == "__main__":
    main()