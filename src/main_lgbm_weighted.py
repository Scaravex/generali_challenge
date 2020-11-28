# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 09:24:19 2020

@author: Marco
"""

import os
import pandas as pd
import gc
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.model_selection import KFold
from datetime import datetime
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

from utils import (
    reduce_mem_usage,
    sub_template_creator,
    prepare_train_test_before_scoring,
    fetch_data,
    subset_data,
    submission_generator,
    find_threshold_cv,
    set_interval_proba,
    get_covariate_weights,
    lasso_ftrs_w,
)

current_dir = os.getcwd()
main_path = os.path.dirname(current_dir)
# main_path =  r'C:\Users\Marco\Documents\GitHub\axa_challenge\src'
os.chdir(main_path)

gc.collect()

# Importing data
training = fetch_data("train")
test = fetch_data("validation")

training = reduce_mem_usage(training)
test = reduce_mem_usage(test)

# Eseguo subset del dataset di train
training = subset_data(training, "random", prcn=1)

print("train shape: ", training.shape, " - test shape: ", test.shape)

# defining predictions dataframe
submission_template = sub_template_creator(test)

X_train, y_train, X_test = prepare_train_test_before_scoring(training, test)

# Fare funzione
# 10 fold cross-validation
n_fold = 10  # mettere 10 alla fine--> per ora ok 5
smote_ratio = 0.25
use_weights = True

folds = KFold(n_fold)
ROC_Avg = 0
prc_avg = 0
submission = submission_template.copy()
start_time = datetime.now()
mdl_list = []

for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train)):
    print(fold_n)
    now = datetime.now()

    X_train_, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_train_, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]

    y_train_ = y_train_.astype("bool")
    y_valid = y_valid.astype("bool")

    if smote_ratio > 0:
        oversample = SMOTE(sampling_strategy=smote_ratio)
        X_train_, y_train_ = oversample.fit_resample(X_train_, y_train_)

    if use_weights:
        lgbm_weights = get_covariate_weights(X_train_, lasso_ftrs_w)
        lgbm_X = lgb.Dataset(data=X_train_, label=y_train_, weight=lgbm_weights,)
    else:
        lgbm_X = lgb.Dataset(data=X_train_, label=y_train_,)

    from models import lgbm_prm

    pred_model = lgb.train(lgbm_prm, lgbm_X, num_boost_round=512)

    pred = pred_model.predict(X_test)
    val = pred_model.predict(X_valid)

    # Make sure that probabilities are between 0 and 1
    pred = set_interval_proba(pred)
    val = set_interval_proba(val)

    print("finish pred")

    # Calculate precision on validation set
    sub_cv = submission.copy()
    sub_cv["y_test_pred"] = pred.copy()
    fold_threshold = find_threshold_cv(sub_cv)
    X_valid["pred_proba"] = val.copy()
    X_valid["pred_label"] = X_valid["pred_proba"].apply(
        lambda x: 1 if (x >= fold_threshold) else 0
    )
    prc = precision_score(y_valid, X_valid["pred_label"])

    # Calculate ROC AUC on validation set
    ROC_auc = roc_auc_score(y_valid, val)

    del X_valid
    print("ROC accuracy: {}".format(ROC_auc))
    print("Precision: {}".format(prc))
    ROC_Avg = ROC_Avg + ROC_auc
    prc_avg = prc_avg + prc
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
print("\nAverage precision is: ", prc_avg / n_fold)
print("Total time to train the models is: ", current_time)


threshold, sub_copy = submission_generator(submission, "lightgbm", mdl_list)
