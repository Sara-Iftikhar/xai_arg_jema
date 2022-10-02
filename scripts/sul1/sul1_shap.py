"""
=====================
shap
=====================
"""


import shap
import pandas as pd
from ai4water import Model
import matplotlib.pyplot as plt
from ai4water.postprocessing import ShapExplainer
shap.__version__

from sul1_utils import get_fitted_model

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

# %%

class MyShapExplainer(ShapExplainer):

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

