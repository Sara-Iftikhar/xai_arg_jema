"""
=======================
aac shap
=======================
"""

# %%
# model building
# ---------------

import numpy as np
import pandas as pd
from ai4water import Model
import matplotlib.pyplot as plt


np.set_printoptions(suppress=True, linewidth=200)

# %%

model = Model(
    model= {
            "CatBoostRegressor": {
                "iterations": 500,
                "learning_rate": 0.49999999999999994,
                "l2_leaf_reg": 0.5,
                "model_size_reg": 3.1912231399066187,
                "rsm": 0.8001459176683743,
                "border_count": 1032,
                "feature_border_type": "UniformAndQuantiles",
                "logging_level": "Silent",
                "random_seed": 891
            }
        },
    x_transformation=[
            {
                "method": "quantile",
                "features": [
                    "wat_temp_c"
                ]
            },
            {
                "method": "robust",
                "features": [
                    "tide_cm"
                ]
            },
            {
                "method": "log",
                "features": [
                    "sal_psu"
                ],
                "treat_negatives": True,
                "replace_zeros": True
            },
            {
                "method": "sqrt",
                "features": [
                    "pcp_mm"
                ],
                "treat_negatives": True
            },
            {
                "method": "scale",
                "features": [
                    "wind_speed_mps"
                ]
            },
            {
                "method": "quantile",
                "features": [
                    "air_p_hpa"
                ]
            }
        ],
    y_transformation=[
            {
                "method": "log",
                "features": [
                    "aac_coppml"
                ],
                "treat_negatives": True,
                "replace_zeros": True
            }
        ],
    seed=891,
    input_features=[
            "wat_temp_c",
            "tide_cm",
            "sal_psu",
            "pcp_mm",
            "wind_speed_mps",
            "air_p_hpa"
        ],
    output_features=[
            "aac_coppml"
        ],
)

# %%
train_df = pd.read_csv("../train_aac_rand.csv", index_col="Unnamed: 0")
train_x, train_y = train_df.iloc[:, 0:-1], train_df.iloc[:, -1]

print(train_x.shape, train_y.shape)

# %%
_ = model.fit(x=train_x.values, y=train_y.values)

# %%
test_df = pd.read_csv("../test_aac_rand.csv", index_col="Unnamed: 0")
test_x, test_y = test_df.iloc[:, 0:-1], test_df.iloc[:, -1]
print(test_x.shape, test_y.shape)

# %%
from ai4water.postprocessing.explain import ShapExplainer
import shap
print(shap.__version__)

#%%

class MyShapExpaliner(ShapExplainer):
    def summary_plot(
            self,
            plot_type: str = None,
            name: str = "summary_plot",
            **kwargs
    ):
        """
        Plots the `summary <https://shap-lrjball.readthedocs.io/en/latest/generated/shap.summary_plot.html#shap.summary_plot>`_
        plot of SHAP package.
        Arguments:
            plot_type : str,
                either "bar", or "violen" or "dot"
            name:
                name of saved file
            kwargs:
                any keyword arguments to shap.summary_plot
        """

        def _summary_plot(_shap_val, _data, _features, _name):
            plt.close('all')

            shap.summary_plot(_shap_val, _data, show=False, plot_type=plot_type,
                              feature_names=_features,
                              **kwargs)
            # if self.save:
            #     plt.savefig(os.path.join(self.path, _name + " _bar"), dpi=300,
            #                 bbox_inches="tight")
            # if self.show:
            #     plt.show()

            return

        shap_vals = self.shap_values
        if isinstance(shap_vals, list) and len(shap_vals) == 1:
            shap_vals = shap_vals[0]

        data = self.data

        if self.single_source:
            if data.ndim == 3:
                assert shap_vals.ndim == 3

                for lookback in range(data.shape[1]):

                    _summary_plot(shap_vals[:, lookback],
                                  _data=data[:, lookback],
                                  _features=self.features,
                                  _name=f"{name}_{lookback}")
            else:
                _summary_plot(_shap_val=shap_vals, _data=data,
                              _features=self.features, _name=name)
        else:
            # data is a list of data sources
            for idx, _data in enumerate(data):
                if _data.ndim == 3:
                    for lb in range(_data.shape[1]):
                        _summary_plot(_shap_val=shap_vals[idx][:, lb],
                                      _data=_data[:, lb],
                                      _features=self.features[idx],
                                      _name=f"{name}_{idx}_{lb}")
                else:
                    _summary_plot(_shap_val=shap_vals[idx], _data=_data,
                                  _features=self.features[idx], _name=f"{name}_{idx}")

        return

# %%

explainer = MyShapExpaliner(model=model,
                         data=test_x,
                         train_data=train_x,
                         feature_names=model.input_features,
                          #save=False
                         )
print(explainer.explainer)

#%%

explainer.plot_shap_values()

#%%
#
# explainer.waterfall_plot_single_example(27)
#
# #%%
#
# explainer.waterfall_plot_single_example(28)
#
# #%%
#
# explainer.waterfall_plot_single_example(29)
#
# #%%
#
# explainer.waterfall_plot_single_example(30)
#
# #%%
#
# explainer.waterfall_plot_single_example(31)
#
# #%%
#
# explainer.waterfall_plot_single_example(32)
#
# #%%
#
# explainer.waterfall_plot_single_example(33)

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
# ------------------

#%%

explainer = MyShapExpaliner(model=model,
                         data=test_x,
                         train_data=train_x,
                         feature_names=model.input_features,
                          explainer="KernelExplainer",
                         # save=False
                         )
print(explainer.explainer)

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
