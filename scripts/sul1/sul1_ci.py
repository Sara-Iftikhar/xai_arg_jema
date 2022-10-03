"""
===================
conformal analysis
===================
"""

import pandas as pd

from sul1_utils import get_fitted_model, confidenc_interval, Model

train_df = pd.read_csv("../train_sul1_rand.csv", index_col="Unnamed: 0")
print(train_df.shape)

# %%
train_df.head()

# %%
X_train, y_train = train_df.iloc[:, 0:-1].values, train_df.iloc[:, -1].values

# %%
test_df = pd.read_csv("../test_sul1_rand.csv", index_col="Unnamed: 0")

print(test_df.shape)

# %%
X_test, y_test = test_df.iloc[:, 0:-1], test_df.iloc[:, -1]

# # %%
model = get_fitted_model(Model)


confidenc_interval(model, X_train, y_train, X_test, alpha=0.05, n_splits=5)

confidenc_interval(model, X_train, y_train, X_test, alpha=0.1, n_splits=5)

confidenc_interval(model, X_train, y_train, X_test, alpha=0.2, n_splits=5)

confidenc_interval(model, X_train, y_train, X_test, alpha=0.3, n_splits=5)