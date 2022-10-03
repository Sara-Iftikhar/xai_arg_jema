"""
===============
aac lime
===============
"""

import pandas as pd
from ai4water.postprocessing import LimeExplainer

from aac_utils import get_fitted_model

# %%
model = get_fitted_model()

# %%
train_df = pd.read_csv("../train_aac_rand.csv", index_col="Unnamed: 0")
train_x, train_y = train_df.iloc[:, 0:-1], train_df.iloc[:, -1]

print(train_x.shape, train_y.shape)

# %%
test_df = pd.read_csv("../test_aac_rand.csv", index_col="Unnamed: 0")
test_x, test_y = test_df.iloc[:, 0:-1], test_df.iloc[:, -1]
print(test_x.shape, test_y.shape)

#%%

explainer = LimeExplainer(
    model=model,
    data=test_x,
    train_data=train_x,
    mode="regression",
    feature_names=model.input_features,
    #save=False
)

#%%

_ = explainer.explain_example(27)

#%%

_ = explainer.explain_example(28)

#%%

_ = explainer.explain_example(29)

#%%

_ = explainer.explain_example(30)

#%%

_ = explainer.explain_example(31)

#%%

_ = explainer.explain_example(32)

#%%

_ = explainer.explain_example(33)