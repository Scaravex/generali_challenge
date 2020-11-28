"""Collection of utilities functions."""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle


def fetch_data(d_type):
    """Return train or validation data as pandas dataframe."""
    if d_type == "train":

        return pd.read_csv("./data/training.csv")
    if d_type == "train_engineered":
        return pd.read_csv("./data/training_fe.csv")
    if d_type == "validation":
        return pd.read_csv("./data/validation.csv")
    if d_type == "validation_engineered":
        return pd.read_csv("./data/validation_fe.csv")
    if d_type == "sample":
        return pd.read_csv("./data/samples/sample_1000.csv")
    if d_type not in [
        "train",
        "train_engineered",
        "validation",
        "validation_engineered",
        "sample",
    ]:
        print("Enter a valid data type.")


def drop_columns_without_variability(df):
    """Drop from df columns that have no variability in training data. Can be applied to train and test."""
    from config import columns_without_variablity

    for col in columns_without_variablity:
        if col in df:
            df.drop(columns=[col], inplace=True)
    return df


def drop_artificial_columns(df):
    """Drop from df columns that have no human meaning."""
    from config import artificial_columns

    for col in artificial_columns:
        if col in df:
            df.drop(columns=[col], inplace=True)
    return df


def reverse_dummies_to_categories(df):
    """Transform dummy variables back into categorical ones."""
    from config import prefixes_list

    for prefix in prefixes_list:
        # print(f"Working with {prefix}")

        # Get a list of all the columns with given prefix
        col_of_interest = []
        for col in df.columns:
            if col.find(prefix + "__") == 0:
                col_of_interest.append(col)

        # Work on a dataframe with only needed colmns
        df_wrk = df[col_of_interest].copy()

        # Rename the columns taking away the prefix
        for col in df_wrk:
            df_wrk[col] = pd.to_numeric(df_wrk[col])
            new_name = col[(col.rfind("__") + 2) :]
            df_wrk.rename(columns={col: new_name}, inplace=True)

        # Create categorical column and drop dummies
        df[prefix] = df_wrk.idxmax(axis=1)
        df.drop(columns=col_of_interest, inplace=True)

    return df


def reduce_mem_usage(df):
    """Iterate through all the columns of a dataframe and modify the data type to reduce memory usage."""
    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        # else:
        # df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print(
        "Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)".format(
            start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem
        )
    )
    return df


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (
                corr_matrix.columns[j] not in col_corr
            ):
                colname = corr_matrix.columns[i]  # getting the name of column
                print(
                    colname,
                    " correlated with",
                    corr_matrix.columns[j],
                    " corr: ",
                    corr_matrix.iloc[i, j],
                )
                col_corr.add(colname)
                # if colname in dataset.columns:
                # del dataset[colname] # deleting the column from the dataset
    return col_corr


def getDummies(dfz, col, minCtn = 10):
    '''
    function which create dummy variables 
    for the different categories
    '''    
    df2 = dfz.copy()
    df2['_id'] = 1
    df_aux = df2.groupby(col).aggregate({'_id':'count'}).reset_index() 
    df_aux = df_aux[df_aux._id>=minCtn]
    topColTypes = list(set(df_aux[col].values))
    dfz[col] = dfz.apply(lambda r: r[col] if r[col] in topColTypes else 'OTHER' , axis=1)
    dummies = pd.get_dummies(dfz[col], prefix=col) # +'_')
    
    return dummies, topColTypes

def preprocess_eda(df):
    """Esegue pre processing necessario a EDA."""

    df = reverse_dummies_to_categories(df)
    df = drop_columns_without_variability(df)
    df = drop_artificial_columns(df)
    if "target_class" in df:
        df.target_class = df.target_class.apply(
            lambda x: "frode" if x == True else "lecito"
        )
        df.rename(columns={"target_class": "tipo_sinistro"}, inplace=True)
    df.diff_days_claim_date_notif_date = df.diff_days_claim_date_notif_date.abs()
    df.diff_days_claim_date_policy_end_date = (
        df.diff_days_claim_date_policy_end_date * (-1)
    )
    print(df.shape)
    return df

def calculate_diff_age(df):
    df["diff_year_fp_tp"] = abs(
        df["diff_year_now_tp__date_of_birth"] - df["diff_year_now_fp__date_of_birth"]
    )
    return df

    
def calcola_complessita_sinistro(df):
    """Assegna il punteggio di complessita al sinistro."""

    df["complessita_sinistro"] = 0
    df["complessita_sinistro"] = df.apply(
        lambda row: row.complessita_sinistro + 1
        if row.vehicle_is_damaged__sum > 2
        else row.complessita_sinistro,
        axis=1,
    )
    df["complessita_sinistro"] = df.apply(
        lambda row: row.complessita_sinistro + 1
        if row.tp__vehicle_is_damaged == True
        else row.complessita_sinistro,
        axis=1,
    )
    df["complessita_sinistro"] = df.apply(
        lambda row: row.complessita_sinistro + 1
        if row.fp__vehicle_is_damaged == True
        else row.complessita_sinistro,
        axis=1,
    )
    df["complessita_sinistro"] = df.apply(
        lambda row: row.complessita_sinistro + 1
        if row.is_police_report == True
        else row.complessita_sinistro,
        axis=1,
    )
    df["complessita_sinistro"] = df.apply(
        lambda row: row.complessita_sinistro + 1
        if row.is_witness == True
        else row.complessita_sinistro,
        axis=1,
    )
    return df


def subset_data(df, shuffle_type, prcn=1, smote_os=0):
    """Decide what type of shuffle is applied to data, then oversamples minority class to hit ratio specified in smote_os."""

    if smote_os > 0:
        oversample = SMOTE(sampling_strategy=smote_os)
        X, y = oversample.fit_resample(
            df.drop(["claim_id", "target_class"], axis=1), df.target_class
        )
        df = X.copy()
        df["target_class"] = y

    if shuffle_type == "last":
        drop_train_idx = (
            df[df["target_class"] != 1]
            .loc[
                : int(len(df["target_class"]))
                - int(len(df[df["target_class"] != 1]) / prcn)
            ]
            .index
        )
        df = df.drop(df.index[drop_train_idx])

    if shuffle_type == "first":
        drop_train_idx = (
            df[df["target_class"] != 1]
            .loc[int(len(df[df["target_class"] != 1]) / prcn) :]
            .index
        )
        df = df.drop(df.index[drop_train_idx])

    if shuffle_type == "random":
        df = df.sample(int(len(df) * prcn))

    return df


def sub_template_creator(df):
    """Crea il template per il file di submission."""
    submission_template = pd.DataFrame(index=range(len(df)))
    submission_template["y_id"] = df["claim_id"]
    submission_template["y_test_pred"] = 0

    return submission_template


def find_threshold_cv(submission):
    """Find the threshold above which observations are predicted as frauds. Use during cross validation."""

    sub_copy = submission.copy()
    sub_copy = sub_copy.sort_values(["y_test_pred"])
    length_test = len(sub_copy["y_test_pred"])
    # aggiunto penalty score
    target_preds = 0.005 * length_test
    n_preds = round(length_test - target_preds)
    threshold = sub_copy.iloc[n_preds]["y_test_pred"]

    return threshold


def submission_generator(submission, model_type, mdl_list):
    """Genera un file pronto per la submission."""

    ref_dt = (
        str(datetime.now())[0:19].replace(" ", "_").replace("-", "").replace(":", "")
    )
    sub_copy = submission.copy()
    sub_copy = sub_copy.sort_values(["y_test_pred"])
    length_test = len(sub_copy["y_test_pred"])
    # aggiunto penalty score
    target_preds = 0.005 * length_test
    n_preds = round(length_test - target_preds)
    threshold = sub_copy.iloc[n_preds]["y_test_pred"]
    del sub_copy
    sub_copy = submission.copy()
    print("The values over this threshold are predicted as frauds:", threshold)
    sub_copy["pred_proba"] = sub_copy["y_test_pred"]
    sub_copy["y_test_pred"] = sub_copy["y_test_pred"].apply(
        lambda x: 1 if (x >= threshold) else 0
    )
    TPFP = sub_copy["y_test_pred"].sum()
    penalty = abs(TPFP - target_preds) / target_preds
    print("Penalty is:", penalty)
    sub_copy.to_csv(
        f"./prediction/with_proba/submission_{ref_dt}.csv", index=False, header=False
    )
    sub_copy.drop(columns=["pred_proba"]).to_csv(
        f"./prediction/submissions/submission_{ref_dt}.csv", index=False, header=False
    )

    if model_type == "lightgbm":
        os.mkdir(f"./prediction/models/lgbm_{ref_dt}")
        i = 0
        for mdl in mdl_list:
            with open(
                f"./prediction/models/lgbm_{ref_dt}/lgbm_{str(i)}.pkl", "wb"
            ) as pickle_file:
                pickle.dump(mdl, pickle_file)
            i = i + 1
    return threshold, sub_copy


def prepare_train_test_before_scoring(df, test):
    """Prepara a livello formale train e test set, prima di fare addestramento e scoring."""

    training = df.copy()
    labels = training["target_class"]  # 1.2%
    test = test.drop(["claim_id"], axis=1)
    for col in ["claim_id", "target_class"]:
        if col in training.columns:
            training.drop(columns=[col], inplace=True)

    X_train, y_train = training, labels
    X_test = test

    X_train.columns = [
        "".join(c if c.isalnum() else "_" for c in str(x)) for x in X_train.columns
    ]
    X_test.columns = [
        "".join(c if c.isalnum() else "_" for c in str(x)) for x in X_test.columns
    ]
    return X_train, y_train, X_test


def set_interval_proba(p_vect):
    """Make sure that probabilities are between 0 and 1."""
    v = p_vect.copy()
    v[v > 1] = 1
    v[v < 0] = 0
    return v


def get_covariate_weights(X, ftrs, normalize=True):
    """Returns probability for each observation to belong to training set."""

    df = X[ftrs].copy()

    # Pre allocate vector with predictions
    df["pred_weights"] = 0

    # For each model, get the prediction
    from os import listdir
    from os.path import isfile, join

    models = [
        f
        for f in listdir(f"./src/weights_mdls/")
        if isfile(join(f"./src/weights_mdls/", f))
    ]
    for mdl in models:
        # Get the model
        with open(f"./src/weights_mdls/{mdl}", "rb") as pickle_file:
            this_mdl = pickle.load(pickle_file)
            # Get the prediction
        prd = this_mdl.predict_proba(df[ftrs])[:, 1]
        df["pred_weights"] = prd + df["pred_weights"]

    # Assigns probabilities
    df["pred_weights"] = df["pred_weights"] / len(models)

    if normalize == False:
        return df["pred_weights"]
    else:
        weights = (1.0 / df["pred_weights"]) - 1.0
        weights /= np.mean(weights)
        return weights


lasso_ftrs_w = [
    "business_rule_20",
    "business_rule_7",
    "business_type__commercial",
    "cid_vehicles_number",
    "claim_amount_category",
    "claim_type_desc__md_rca_cid_misto",
    "claim_type_desc__pa_ard_eventi_speciali",
    "client_responsibility",
    "coverage__responsabilita_civile_auto",
    "coverage_excess_amount__sum",
    "coverage_insured_amount__sum",
    "diff_days_claim_date_notif_date",
    "diff_days_claim_date_original_start_date",
    "diff_days_claim_date_policy_end_date",
    "diff_days_claim_date_policy_start_date",
    "diff_year_now_fp__date_of_birth",
    "diff_year_now_tp__date_of_birth",
    "dist_claim_fp",
    "dist_claim_tp",
    "dist_fp_tp",
    "driving_licence_type__other",
    "fp__vehicle_type__car",
    "fp__vehicle_type__truck",
    "insured_item_code__none",
    "insured_item_unit_section__fur4a",
    "insured_item_unit_section__other",
    "insured_item_unit_section__rca",
    "insured_item_unit_section__rcap",
    "insured_value2__sum_category",
    "network_feature_20",
    "network_feature_25",
    "network_feature_26",
    "network_feature_33",
    "network_feature_35",
    "network_feature_42",
    "party_type__105",
    "party_type__other",
    "policy_branch__none",
    "policy_broker_code__none",
    "policy_premium_amount_category",
    "policy_status__11",
    "policy_status__none",
    "policy_status__other",
    "policy_transaction_description__none",
    "region_of_claim__lombardia",
    "risk_code__apfu4a",
    "risk_code__pvrca",
    "tarif_type__bonusmalus",
    "tarif_type__none",
    "total_reserved_category",
    "vehicle_is_damaged__sum",
]
