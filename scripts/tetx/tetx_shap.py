"""
=====================
shap
=====================
"""

import os

from shap import Explanation

import shap
import pandas as pd
from ai4water import Model
import matplotlib.pyplot as plt
from ai4water.postprocessing import ShapExplainer
shap.__version__

from tetx_utils import get_fitted_model, tetx_data

# %%
model = get_fitted_model(Model)

# %%

x, y, input_features, output_features = tetx_data()

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

    def waterfall_plot_single_example(
            self,
            example_index: int,
            name: str = "waterfall",
            max_display: int = 10,
    ):
        """draws and saves waterfall_ plot
         for one example.
        The waterfall plots are based upon SHAP values and show the
        contribution by each feature in model's prediction. It shows which
        feature pushed the prediction in which direction. They answer the
        question, why the ML model simply did not predict mean of training y
        instead of what it predicted. The mean of training observations that
        the ML model saw during training is called base value or expected value.
        Arguments:
            example_index : int
                index of example to use
            max_display : int
                maximu features to display
            name : str
                name of plot
        .. _waterfall:
            https://shap.readthedocs.io/en/latest/generated/shap.plots.waterfall.html
        """
        if self.explainer.__class__.__name__ in ["Deep", "Kernel"]:
            shap_vals_as_exp = None
        else:
            shap_vals_as_exp = self.explainer(self.data)

        shap_values = self.shap_values
        if isinstance(shap_values, list) and len(shap_values) == 1:
            shap_values = shap_values[0]

        plt.close('all')

        if shap_vals_as_exp is None:

            features = self.features
            if not self.data_is_2d:
                features = self.unrolled_features

            # waterfall plot expects first argument as Explaination class
            # which must have at least these attributes (values, data, feature_names, base_values)
            # https://github.com/slundberg/shap/issues/1420#issuecomment-715190610
            if not self.data_is_2d:  # if original data is 3d then we flat it into 1d array
                values = shap_values[example_index].reshape(-1, )
                data = self.data[example_index].reshape(-1, )
            else:
                values = shap_values[example_index]
                data = self.data.iloc[example_index]

            exp_value = self.explainer.expected_value
            if self.explainer.__class__.__name__ in ["Kernel"]:
                pass
            else:
                exp_value = exp_value[0]

            e = Explanation(
                values,
                base_values=exp_value,
                data=data,
                feature_names=features
            )

            shap.plots.waterfall(e, show=False, max_display=max_display)
        else:
            shap.plots.waterfall(shap_vals_as_exp[example_index], show=False, max_display=max_display)

        if self.save:
            plt.savefig(os.path.join(self.path, f"{name}_{example_index}"),
                        dpi=300,
                        bbox_inches="tight")

        if self.show:
            plt.show()

        return

#%%

explainer = MyShapExplainer(model=model,
                         data=x,
                         train_data=x,
                         feature_names=model.input_features,
                          #save=False
                         )
explainer.explainer

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

# explainer.scatter_plot_all_features()

#%% md

## Kernel Explainer

#%%

explainer = MyShapExplainer(model=model,
                         data=x,
                         train_data=x,
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

