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
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
def main():
    
# =============================================================================
#       Data load, no default {80368} : default {19632}
# =============================================================================
    
    df = pd.read_excel(".\PastLoans.xlsx") #base_df   
    df.drop('employment', inplace=True, axis=1)
    
    
# =============================================================================
#       Data Preperation for all 3 socials
# =============================================================================
    
    X = df.iloc[:, 0:9].values
    Y = df.iloc[:, 9].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(X_train[0:3, :])
    
# =============================================================================
#       Data is clean, saving computation time
# =============================================================================

    # X_train_clean = X_train[~np.isnan(X_train).any(axis=1)]
    # to_delete = []
    # for i in range(len(X_train)):
    #     for j in range(len(X_train[i])):
    #         if np.isnan(X_train[i][j]):
    #             to_delete.append(i)
    # Y_train_clean = np.delete(Y_train, to_delete, axis = 0)
    # X_test_clean = X_test[~np.isnan(X_test).any(axis=1)]
    # to_delete1 = []
    # for i in range(len(X_test)):
    #     for j in range(len(X_test[i])):
    #         if np.isnan(X_test[i][j]):
    #             to_delete1.append(i)
    # Y_test_clean = np.delete(Y_test, to_delete1, axis = 0)
    
# =============================================================================
#       Data Preperation for social1
# =============================================================================

    df2 = df
    df2.drop('social2', inplace=True, axis=1)
    df2.drop('social3', inplace=True, axis=1) #lazy at 2AM
    X2 = df2.iloc[:, 0:7].values
    Y2 = df2.iloc[:, 7].values
    X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y2, test_size=0.1, random_state=42,stratify=Y2)
    sc = MinMaxScaler()
    X_train2 = sc.fit_transform(X_train2)
    X_test2 = sc.transform(X_test2)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(X_train[0:3, :])
    
# =============================================================================
# BalancedRandomForestClassifier(class_weight='balanced_subsample', max_depth=10,
                               # min_samples_leaf=50, n_estimators=200, n_jobs=-1,
                               # random_state=42) as best result using GridSearchCV    

# RandomForestClassifier(max_depth=10, min_samples_leaf=100, n_jobs=-1,
                       # random_state=42) as best result using GridSearchCV
# =============================================================================
    
#     rf = BalancedRandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced_subsample')
    
#     params = {
#     'max_depth': [2,3,5,10,20],
#     'min_samples_leaf': [5,10,20,50,100,200],
#     'n_estimators': [10,25,30,50,100,200]
# }
#     grid_search = GridSearchCV(estimator=rf,
#                             param_grid=params,
#                             cv = 4,
#                             n_jobs=-1, verbose=1, scoring="roc_auc")
#     grid_search.fit(X_train2, Y_train2)
#     print(grid_search.best_score_)
#     rf_best = grid_search.best_estimator_
#     print(rf_best)

# =============================================================================
#     Classifier using best results from above + all 3 socials
# =============================================================================

#     classifier = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf= 100)
#     classifier.fit(X_train, Y_train)
    
# # =============================================================================
# #     Results
# # =============================================================================

#     imp_df = pd.DataFrame({
#     "Varname": df.columns.drop('default'),
#     "Importance": classifier.feature_importances_
# })
#     print(imp_df.sort_values(by=["Importance"], ascending=False))
#     Y_pred = classifier.predict(X_test)
#     # print(confusion_matrix(Y_test,Y_pred))
#     fig1 = plot_confusion_matrix(classifier, X_test, Y_test, display_labels=['Will Not Default', 'Will Default'], cmap='Greens')
#     plt.title('Standard Random Forest Confusion Matrix')
#     plt.show()
#     print("Accuracy score for model with all 3 socials is " + str(accuracy_score(Y_test, Y_pred)*100)+"%")
#     print('ROCAUC score:',roc_auc_score(Y_test, Y_pred))
#     print('F1 score:',f1_score(Y_test, Y_pred))


# =============================================================================
#     Classifier using best results from above + social1 only
# =============================================================================

#     classifier2 = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf= 100)
#     classifier2.fit(X_train2, Y_train2)
    
# # =============================================================================
# #     Results
# # =============================================================================

#     imp_df = pd.DataFrame({
#     "Varname": df2.columns.drop('default'),
#     "Importance": classifier2.feature_importances_
# })
#     print(imp_df.sort_values(by=["Importance"], ascending=False))
#     Y_pred2 = classifier2.predict(X_test2)
#     # print(confusion_matrix(Y_test2,Y_pred2))
#     fig2 = plot_confusion_matrix(classifier2, X_test2, Y_test2, display_labels=['Will Not Default', 'Will Default'], cmap='Greens')
#     plt.title('Standard Random Forest Confusion Matrix')
#     plt.show()
#     print("Accuracy score for social1 model is " + str(accuracy_score(Y_test2, Y_pred2)*100)+"%") 
#     print('ROCAUC score:',roc_auc_score(Y_test2, Y_pred2))
#     print('F1 score:',f1_score(Y_test2, Y_pred2))
    
# =============================================================================
#     Balanced Classifier using best results from above + social1 only
# =============================================================================

    classifier3 = BalancedRandomForestClassifier(max_depth=10, min_samples_leaf=50, n_estimators=200, n_jobs=-1, random_state=42, class_weight='balanced_subsample')
    classifier3.fit(X_train2, Y_train2)
    
# =============================================================================
#     Results
# =============================================================================
    imp_df = pd.DataFrame({
    "Varname": df2.columns.drop('default'),
    "Importance": classifier3.feature_importances_
})
    print(imp_df.sort_values(by=["Importance"], ascending=False))
    Y_pred3 = classifier3.predict(X_test2)
    # print(confusion_matrix(Y_test2,Y_pred3))
    fig3 = plot_confusion_matrix(classifier3, X_test2, Y_test2, display_labels=['Will Not Default', 'Will Default'], cmap='Greens')
    plt.title('Standard Random Forest Confusion Matrix')
    plt.show()
    # print("Accuracy score for balanced social1 model is " + str(accuracy_score(Y_test2, Y_pred3)*100)+"%")  
    print('ROCAUC score for balanced social1 model is:',roc_auc_score(Y_test2, Y_pred3))
    print('Report:',classification_report(Y_test2, Y_pred3))
    print('F1 score:',f1_score(Y_test2, Y_pred3))
    
# =============================================================================
#     Let's make the new excel
# =============================================================================

    output_df = pd.read_excel(".\LoanApplications_Stage1_Lender1.xlsx")
    output_df.drop('employment', inplace=True, axis=1)
    output_X = output_df.iloc[:, 0:7].values
    sc = MinMaxScaler()
    output_scaled_X = sc.fit_transform(output_X)
    Y_pred4 = classifier3.predict(output_scaled_X)
    Y_pred4_probabilities = classifier3.predict_proba(output_scaled_X)
    output_df['default'] = Y_pred4
    output_df['probability of no default'] = Y_pred4_probabilities[:,0]
    output_df['probability of default'] = Y_pred4_probabilities[:,1]
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(output_df.head())
        
    output_df.to_excel(".\LoanApplications_Stage1_result.xlsx", index = False)
    
# # =============================================================================
# #     Classifier + SMOTE using best results from above + social1 only
# # =============================================================================

#     oversample = SMOTE()
#     over_X2, over_Y2 = oversample.fit_resample(X2, Y2)
#     over_X_train2, over_X_test2, over_Y_train2, over_Y_test2 = train_test_split(over_X2, over_Y2, test_size=0.1, stratify = over_Y2)
#     X_train3, X_test3, Y_train3, Y_test3 = train_test_split(X2, Y2, test_size=0.1, stratify = Y2)
#     classifier4 = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf= 100)
#     classifier4.fit(over_X_train2, over_Y_train2)
    
# # =============================================================================
# #     Results
# # =============================================================================

#     imp_df = pd.DataFrame({
#     "Varname": df2.columns.drop('default'),
#     "Importance": classifier4.feature_importances_
# })
#     print(imp_df.sort_values(by=["Importance"], ascending=False))
#     Y_pred4 = classifier4.predict(X_test3)
#     Y_pred4_probabilities = classifier4.predict_proba(X_test3)
#     print(Y_pred4_probabilities)
#     print(len(Y_pred4_probabilities))
#     # print(confusion_matrix(over_Y_test2,Y_pred4))
#     fig4 = plot_confusion_matrix(classifier4, X_test3, Y_test3, display_labels=['Will Not Default', 'Will Default'], cmap='Greens')
#     plt.title('Standard Random Forest Confusion Matrix')
#     plt.show()
#     # print("Accuracy score for SMOTE social1 model is " + str(accuracy_score(Y_test3, Y_pred4)*100)+"%")
#     print('ROCAUC score for SMOTE social1 model is:',roc_auc_score(Y_test3, Y_pred4))
#     print('Report:',classification_report(Y_test2, Y_pred4))
#     print('F1 score:',f1_score(Y_test3, Y_pred4))

# # =============================================================================
# #        Data Preperation for no socials
# # =============================================================================

#     df3 = df2
#     df3.drop('social1', inplace=True, axis=1)
#     X3 = df3.iloc[:, 0:6].values
#     Y3 = df3.iloc[:, 6].values     
#     X_train3, X_test3, Y_train3, Y_test3 = train_test_split(X3, Y3, test_size=0.1, random_state=42,stratify=Y3)
#     sc = MinMaxScaler()
#     X_train3 = sc.fit_transform(X_train3)
#     X_test3 = sc.transform(X_test3)
    

# # =============================================================================
# #  Balanced Classifier without socials
# # =============================================================================

#     classifier5 = BalancedRandomForestClassifier(max_depth=10, min_samples_leaf=50, n_estimators=200, n_jobs=-1, random_state=42, class_weight='balanced_subsample')
#     classifier5.fit(X_train3, Y_train3)
    
# # =============================================================================
# #     Results
# # =============================================================================

#     imp_df = pd.DataFrame({
#     "Varname": df3.columns.drop('default'),
#     "Importance": classifier5.feature_importances_
# })
#     print(imp_df.sort_values(by=["Importance"], ascending=False))
#     Y_pred5 = classifier5.predict(X_test3)
#     # print(confusion_matrix(Y_test2,Y_pred3))
#     fig5 = plot_confusion_matrix(classifier5, X_test3, Y_test3, display_labels=['Will Not Default', 'Will Default'], cmap='Greens')
#     plt.title('Standard Random Forest Confusion Matrix')
#     plt.show()
#     # print("Accuracy score for balanced social1 model is " + str(accuracy_score(Y_test2, Y_pred3)*100)+"%")  
#     print('ROCAUC score for balanced no social model is:',roc_auc_score(Y_test3, Y_pred5))
#     print('Report:',classification_report(Y_test3, Y_pred5))
#     print('F1 score:',f1_score(Y_test3, Y_pred5))


    
if __name__ == "__main__":
    main()