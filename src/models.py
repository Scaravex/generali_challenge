# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 09:24:19 2020

@author: Marco
"""

# Marco fix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

import lightgbm as lgb

import xgboost as xgb
import catboost

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def model_train(model_type, X_train_, X_valid, y_train_, y_valid):
    """ tree,lightgbm, xgboost, catboost, randomforest, adaboost, logit, knn, gmm, svn, lda, naivebayes """

    if model_type == "tree":
        treeclf = DecisionTreeClassifier(max_depth=7)
        treeclf.fit(X_train_, y_train_)
        pred_model = treeclf
        del treeclf
    if model_type == "bagging":
        bagclf = BaggingClassifier(
            KNeighborsClassifier(), max_samples=0.5, max_features=0.5
        )
        bagclf.fit(X_train_, y_train_)
        pred_model = bagclf
        del bagclf

    if model_type == "lightgbm":
        # 0.8711  --> 22 minuti e 1
        # dtrain = lgb.Dataset(X_train, label=y_train) #,categorical_feature = categorical_columns)
        # dvalid = lgb.Dataset(X_valid, label=y_valid) #,categorical_feature = categorical_columns)
        lgbclf = lgb.LGBMClassifier(
            num_leaves=512,  # was 512 - default 31
            n_estimators=512,  # default 100 was 512
            max_depth=8,  # default -1, was 9
            learning_rate=0.1,  # default 0.1
            feature_fraction=0.4,  # default 1 was 0.4,
            bagging_fraction=0.4,  # default 1 was 0.4, # subsample by row
            metric="auc",  # binary_logloss auc
            boosting_type="gbdt",  # goss # dart --> speed: goss>gbdt>dart
            lambda_l1=0.4,  # default 0 - 0.4
            lambda_l2=0.6,  # default 0 - 0.6
            scale_pos_weight=18,  # defualt 1
        )

        lgbclf.fit(X_train_, y_train_)
        pred_model = lgbclf
        del lgbclf

    elif model_type == "xgboost":
        # sooo slow  #0.8614
        # scale_pos_weight and adjust settings
        # https://stats.stackexchange.com/questions/243207/what-is-the-proper-usage-of-scale-pos-weight-in-xgboost-for-imbalanced-datasets
        xgbclf = xgb.XGBClassifier(
                    num_leaves=512,
                    n_estimators=512,
                    max_depth=25,
                    learning_rate=0.1,
                    feature_fraction=0.4,
                    bagging_fraction=0.4,
                    subsample=0.85,
                    metric="auc",  # binary_logloss
                    colsample_bytree=0.85,
                    boosting_type="gbdt",  # goss # dart --> speed: goss>gbdt>dart
                    reg_alpha=0.4,
                    reg_lamdba=0.6,
                    scale_pos_weight=82.9,
                )
        xgbclf.fit(X_train_, y_train_)
        pred_model = xgbclf
        del xgbclf

    elif model_type == "catboost":
        # serve farlo anche negli altri modelli?
        ycopy = y_train_.copy()
        ycopy["target_class"] = ycopy["target_class"].apply(
            lambda x: 1 if (x >= 0.5) else 0
        )
        X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(
            X_train_, ycopy.values.flatten(), test_size=0.05
        )
        params = {
            "loss_function": "Logloss",  # objective function
            "eval_metric": "AUC",  # metric
            "verbose": 200,  # output to stdout info about training process every 200 iterations
        }
        catclf = catboost.CatBoostClassifier(**params)
        catclf.fit(
            X_train_1,
            y_train_1,  # data to train on (required parameters, unless we provide X as a pool object, will be shown below)
            eval_set=(X_valid_1, y_valid_1),  # data to validate on
            use_best_model=True,  # True if we don't want to save trees created after iteration with the best validation score
            plot=True,  # True for visualization of the training process (it is not shown in a published kernel - try executing this code)
        )

        del X_train_1, X_valid_1, y_train_1, y_valid_1
        pred_model = catclf
        del catclf

    elif model_type == "randomforest":
        # 0.8476
        # che senso associata ad una singola prediction a 1.6???
        # https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76
        rfclf = RandomForestClassifier(
            n_estimators=512, bootstrap=True, max_features="sqrt"
        )

        rfclf.fit(X_train_, y_train_)
        pred_model = rfclf
        del rfclf

    elif model_type == "adaboost":
        # 0.851 -->8:16:45 ore
        # https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781787286382/9/ch09lvl1sec95/tuning-an-adaboost-regressor
        # https://towardsdatascience.com/boosting-algorithm-adaboost-b6737a9ee60c
        adaclf = AdaBoostClassifier(n_estimators=512, learning_rate=0.0069)
        adaclf.fit(X_train_, y_train_)
        pred_model = adaclf
        del adaclf

    elif model_type == "logit":
        # 0.7764 --> 12:52 minuti senza GridSearch, con gridsearch 63.8%
        ## che senso associata ad una singola prediction a 1.6??? con grid search 0.5
        # https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76
        logregclf = LogisticRegression(penalty="l1", solver="saga", tol=1e-3)
        pipe = Pipeline([("model", logregclf)])
        param_grid = {"model__max_iter": [1000]}
        # adding grid_search to logit
        logregclf_cv = GridSearchCV(
            pipe, param_grid=param_grid, scoring="roc_auc", cv=3
        )

        logregclf_cv.fit(X_train_, y_train_)
        # print('best_params_={}\nbest_score_={}'.format(repr(logregclf_cv.best_params_), repr(logregclf_cv.best_score_)))

        logregclf = logregclf_cv.best_estimator_

        pred_model = logregclf

        del logregclf

    elif model_type == "knn":
        # 0.612  Tempo: 3:21:59.613695
        # https://www.quora.com/How-can-I-choose-the-best-K-in-KNN-K-nearest-neighbour-classification
        knnclf = KNeighborsClassifier(n_neighbors=3, leaf_size=30)  # ), 'p': 1})
        knnclf.fit(X_train_, y_train_)
        pred_model = knnclf
        del knnclf

    elif model_type == "gmm":
        # 0  Tempo:
        # https://www.kaggle.com/albertmistu/detect-anomalies-using-gmm
        gmmclf = GaussianMixture()  # gaussian mixture model
        ycopy = y_train_.copy()
        ycopy["target_class"] = ycopy["target_class"].apply(
            lambda x: 1 if (x >= 0.5) else 0
        )
        gmmclf.fit(X_train_, ycopy)
        pred_model = gmmclf
        del gmmclf
    elif model_type == "svm":
        # 0  Tempo:
        # https://www.kaggle.com/kojr1234/fraud-detection-using-svm
        svcclf = SVC(kernel="rbf", gamma=4 * 1e-3, C=10)
        svcclf.fit(X_train_, y_train_)
        pred_model = svcclf
        del svcclf
    elif model_type == "lda":
        # 0  Tempo:
        ldaclf = LinearDiscriminantAnalysis()
        ldaclf.fit(X_train_, y_train_)
        pred_model = ldaclf
        del ldaclf
    elif model_type == "naivebayes":
        # 0  Tempo:
        gnbclf = GaussianNB()  # priors = [0.995,0.005])
        gnbclf.fit(X_train_, y_train_)
        pred_model = gnbclf
        del gnbclf

    else:
        print("Please, try one of the possible models")

    del X_train_, y_train_
    print("finish train")

    return pred_model, X_valid.copy(), y_valid.copy()


lgbm_prm = {
    "num_leaves": 512,  # was 512 - default 31
    "max_depth": -1,  # default -1, was 9
    "learning_rate": 0.1,  # default 0.1
    "feature_fraction": 0.4,  # default 1 was 0.4,
    "bagging_fraction": 0.4,  # default 1 was 0.4, # subsample by row
    "metric": "auc",  # binary_logloss auc
    "boosting_type": "gbdt",  # goss # dart --> speed: goss>gbdt>dart
    "lambda_l1": 0.4,  # default 0 - 0.4
    "lambda_l2": 0.6,  # default 0 - 0.6
    "scale_pos_weight": 1,  # defualt 1
}

