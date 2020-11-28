# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 09:24:19 2020

@author: Marco
"""

import os
import pandas as pd
import seaborn as sns
import gc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix  # da implementare
from datetime import datetime


from src.models import model_train
from src.utils import (
    reduce_mem_usage,
    sub_template_creator,
    prepare_train_test_before_scoring,
    fetch_data,
    subset_data,
    submission_generator,
    set_interval_proba,
)

current_dir = os.getcwd()
main_path = os.path.dirname(current_dir)
# main_path =  r'C:\Users\Marco\Documents\GitHub\axa_challenge'
os.chdir(main_path)

gc.collect()

# Importing data
training = fetch_data("train_engineered")
test = fetch_data("validation_engineered")

training = reduce_mem_usage(training)
test = reduce_mem_usage(test)

# Eseguo subset del dataset di train
training = subset_data(training, "random", prcn=1, smote_os=0)

training = subset_data(training, "random", 1)
print("train shape: ", training.shape, " - test shape: ", test.shape)

# defining predictions dataframe
submission_template = sub_template_creator(test)


X_train, y_train, X_test = prepare_train_test_before_scoring(training, test)

# Fare funzione
# 10 fold cross-validation
n_fold = 10  # mettere 10 alla fine--> per ora ok 5

folds = KFold(n_fold)
ROC_Avg = 0
submission = submission_template.copy()
start_time = datetime.now()
mdl_list = []

for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train)):
    print(fold_n)
    now = datetime.now()

    X_train_, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_train_, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]


    model_type = "xgboost"  # "lightgbm"

    pred_model, X_valid, y_valid = model_train(
        model_type, X_train_, X_valid, y_train_, y_valid
    )

    pred = pred_model.predict_proba(X_test)[:, 1]
    val = pred_model.predict_proba(X_valid)[:, 1]
    print("finish pred")
    del X_valid
    ROC_auc = roc_auc_score(y_valid, val)
    print("ROC accuracy: {}".format(ROC_auc))
    ROC_Avg = ROC_Avg + ROC_auc
    del val, y_valid
    submission["y_test_pred"] = submission["y_test_pred"] + pred / n_fold
    mdl_list.append(pred_model)
    del pred
    model_time = datetime.now() - now
    total_exp_time = model_time * n_fold
    current_time = datetime.now() - start_time
    print(
        "The current model took in total",
        model_time,
        "\n Still missing, this time:",
        str(total_exp_time - current_time),
    )
    gc.collect()

# 0.87 ldbm-gbdt # 0.86 dart
print("\nAverage ROC is: ", ROC_Avg / n_fold)
print("Total time to train the models is: ", current_time)


threshold, sub_copy = submission_generator(submission, model_type, mdl_list)


####
####
###
# Che ne dici di tenere il main pulito, eliminando tutta la roba qua sotto, e usare un altro script tipo wrk.py per le analisi aggiuntive?


# Idea: using Isolation Forest to improve predictions
# https://medium.com/@Zelros/anomaly-detection-with-t-sne-211857b1cd00

y_train_pred = pred_model.predict_proba(X_train)[:, 1]
temp = pd.DataFrame(data=y_train_pred)
temp[0] = temp[0].apply(lambda x: 1 if (x >= threshold) else 0)
round(temp).sum() / len(
    temp
)  # 7.9% predicted as frauds?? #0.3% 0.29% Boosting# 0.97RandomForest #0.05 Logit
# 0.70% with sampling #2.7% with knn

## why unnamed 0 is so important??????
## le variabili sono ordinate? (e.g. ci sono magari frodi consecutive o in periodi simili dell'anno)

"""
import matplotlib.pyplot as plt
import numpy as np
# Plot feature importance
feature_importance = pred_model.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
sorted_idx = sorted_idx[len(feature_importance) - 50:]
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(10,12))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()




# https://towardsdatascience.com/explainable-artificial-intelligence-part-3-hands-on-machine-learning-model-interpretation-e8ebe5afc608
import eli5
import shap
shap.initjs() 

fig = plt.figure(figsize = (16, 12))
title = fig.suptitle("Default Feature Importances from XGBoost", fontsize=14)

ax1 = fig.add_subplot(2,2, 1)
xgb.plot_importance(pred_model, importance_type='weight', ax=ax1,max_num_features=30)
t=ax1.set_title("Feature Importance - Feature Weight")

ax2 = fig.add_subplot(2,2, 2)
xgb.plot_importance(pred_model, importance_type='gain', ax=ax2,max_num_features=30)
t=ax2.set_title("Feature Importance - Split Mean Gain")

ax3 = fig.add_subplot(2,2, 3)
xgb.plot_importance(pred_model, importance_type='cover', ax=ax3,max_num_features=30)
t=ax3.set_title("Feature Importance - Sample Coverage")

fig = plt.figure(figsize = (16, 12))
from IPython.display import display, HTML
%html
eli5.show_weights(pred_model, top=30)
print(eli5.format_as_html(eli5.explain_weights(pred_model)))




fpr, tpr, thresholds = roc_curve(y_train,y_train_pred)
#ROC Curve
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='*')


# Calculate f1 score
R_f1 = f1_score(y_train, temp[0] )

# Calculate confusion_matrix
conf_matrix = confusion_matrix(y_train,temp[0] )


### only for CART --> plotting tree
from IPython.display import Image  
import pydot
from sklearn.tree import export_graphviz

dot_data = export_graphviz(pred_model, out_file=None,
                filled=True, rounded=True,
                feature_names=X_train.columns,
                special_characters=True)
(graph,) = pydot.graph_from_dot_data(dot_data)
Image(graph.create_png())
"""
