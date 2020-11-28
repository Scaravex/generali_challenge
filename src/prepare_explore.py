# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 09:24:19 2020

@author: Marco
"""

import os
import pandas as pd
import seaborn as sns

current_dir = os.getcwd()
main_path = os.path.dirname(current_dir)
os.chdir(main_path)


def main():
    """Load training and testing data."""
    training = pd.read_csv("../data/dataset/training.csv.gz", compression="gzip")
    # training.head()
    # training.describe()
    test = pd.read_csv("../data/dataset/validation.csv.gz", compression="gzip")

    # 900k training vs 100k test --> target class
    training.to_csv("../data/training.csv", index=False)
    test.to_csv("../data/test.csv", index=False)

    print("training and test csv created")
    return training, test


if __name__ == "__main__":
    training, test = main()


test_column = set(training.columns) - set(test.columns)
training.shape
test.shape

test = test.drop(["claim_id"], axis=1)
trainCorr = training.drop([test_column], axis=1).corr()
testCorr = test.corr()
sns.heatmap(BasicCorr)

#################### SEE/ELIMIANTE CORRELATED FEATURES ######################
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


# Drop features
to_drop = correlation(test, 0.95)
# test.drop(to_drop, axis=1, inplace=True)

labels = training[test_column]  # 1.2%
X_train, y_train = training, labels
X_test = test



from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

# normalizzo le variabili
x = StandardScaler().fit_transform(X_train)

# provo con la pca
pca = PCA(n_components=4, scale=True)
pca_result = pca.fit_transform(x)
pca_df = pd.DataFrame(columns=["pca1", "pca2", "pca3", "pca4"])

pca_df["pca1"] = pca_result[:, 0]
pca_df["pca2"] = pca_result[:, 1]
pca_df["pca3"] = pca_result[:, 2]
pca_df["pca4"] = pca_result[:, 3]
print(
    "Variance explained per principal component: {}".format(
        pca.explained_variance_ratio_
    )
)

pca_df = pd.DataFrame(
    {
        "pca-one": pca_result[:, 0],
        "pca-two": pca_result[:, 1],
        "col": y_train["target_class"],
    }
)

sns.scatterplot(x="pca-one", y="pca-two", hue="col", legend="full", data=pca_df)

# plotting also on test set to see if they are similar or different
x_t = StandardScaler().fit_transform(X_test)
newdata_transformed = pca.transform(x_t)


## i dati non sono messi così bene: non si vedono ad occhio dei clusters
# tsne --> al max 5000 dati o impazzisce
# non sembrano esserci cluster di dati rossi --> metodi tipo Isolation Forest potrebbero non servire
import numpy as np
from sklearn.manifold import TSNE

X_embedded = TSNE(n_components=2).fit_transform(x[90000:95000])
X_embedded.shape
tsne = TSNE()
X_embedded = tsne.fit_transform(x[90000:95000])


tsne_df = pd.DataFrame(
    {
        "X": X_embedded[:, 0],
        "Y": X_embedded[:, 1],
        "col": y_train["target_class"][90000:95000],
    }
)
sns.scatterplot(x="X", y="Y", hue="col", legend="full", data=tsne_df)
