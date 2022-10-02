"""
=====================
shap
=====================
"""


import shap
import pandas as pd
from ai4water import Model
from utils import MyShapExplainer
shap.__version__

from utils import get_fitted_model

# %%
model = get_fitted_model(Model)

# %%
train_df = pd.read_csv("../train_sul1_rand.csv", index_col="Unnamed: 0")
train_x, train_y = train_df.iloc[:, 0:-1], train_df.iloc[:, -1]

print(train_x.shape, train_y.shape)

# %%
test_df = pd.read_csv("../test_sul1_rand.csv", index_col="Unnamed: 0")
test_x, test_y = test_df.iloc[:, 0:-1], test_df.iloc[:, -1]
print(test_x.shape, test_y.shape)


#%%

explainer = MyShapExplainer(model=model,
                         data=test_x,
                         train_data=train_x,
                         feature_names=model.input_features,
                          #save=False
                         )
explainer.explainer

#%%

explainer.plot_shap_values()

#%%

explainer.waterfall_plot_single_example(27)

#%%

explainer.waterfall_plot_single_example(28)

#%%

explainer.waterfall_plot_single_example(29)

#%%

explainer.waterfall_plot_single_example(30)

#%%

explainer.waterfall_plot_single_example(31)

#%%

explainer.waterfall_plot_single_example(32)

#%%

explainer.waterfall_plot_single_example(33)

#%%

#explainer.beeswarm_plot()

#%%

#explainer.heatmap()

#%%

explainer.summary_plot(plot_type="dot")

#%%

explainer.summary_plot(plot_type="bar")

#%%

explainer.summary_plot(plot_type="violin")

#%%

explainer.dependence_plot_all_features()

#%%

explainer.scatter_plot_all_features()

#%% md

## Kernel Explainer

#%%

explainer = MyShapExplainer(model=model,
                         data=test_x,
                         train_data=train_x,
                         feature_names=model.input_features,
                          explainer="KernelExplainer",
                          #save=False
                         )
explainer.explainer

#%%

explainer.plot_shap_values()

#%%

#explainer.beeswarm_plot()

#%%

explainer.summary_plot(plot_type="bar")

#%%

explainer.summary_plot(plot_type="dot")

#%%

explainer.summary_plot(plot_type="violin")

#%%

#explainer.heatmap()

#%%

#explainer.waterfall_plot_single_example(27)
explainer.dependence_plot_all_features()

#%%

#explainer.scatter_plot_all_features()

