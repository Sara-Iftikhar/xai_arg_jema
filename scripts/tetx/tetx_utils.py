import os
import random
import warnings
import importlib
from typing import Union, Callable, List

import numpy as np
import pandas as pd
import shap

import matplotlib.pyplot as plt
from SALib.plotting.hdmr import plot
import easy_mpl as ep

from ai4water import Model as _Model
from ai4water.postprocessing import ShapExplainer
from ai4water.datasets import busan_beach
from ai4water.preprocessing import DataSet
from ai4water.postprocessing._sa import morris_plots
from ai4water.postprocessing.explain._partial_dependence import (compute_bounds,
    _add_dist_as_grid, process_axis)
from ai4water.postprocessing import PartialDependencePlot

from sklearn.model_selection import KFold


def read_data(file_name, inputs= None, target='ecoli',
              power_transform_target=True):

    fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", file_name)
    df = pd.read_excel(fpath, index_col="Date_Time2")
    df.index = pd.to_datetime(df.index)

    default_inputs = ['wat_temp_c', 'tide_cm', 'sal_psu', 'pcp_mm', 'wind_speed_mps', 'air_p_hpa',
    'rel_hum']

    default_targets = [col for col in df.columns if col not in default_inputs]

    if inputs is None:
        inputs = default_inputs

    if not isinstance(target, list):
        if isinstance(target, str):
            target = [target]
    elif isinstance(target, list):
        pass
    else:
        target = default_targets

    assert isinstance(target, list)

    df = df[inputs + target]

    if power_transform_target:
        df[target] = np.power(10, df[target].values)

    return df


def make_whole_data(target,
                    reindex=True,
                    version='new',
                    remove_humidity=True,
                    power_transform_target=True,
                    ):
    data1 = busan_beach(inputs=['wat_temp_c', 'tide_cm', 'sal_psu', 'pcp_mm', 'wind_speed_mps',
                                'air_p_hpa', 'rel_hum'], target=target)

    assert version in ['new', 'old']

    if version == 'new':
        data2 = read_data(target=target, file_name="KarachiData_new.xlsx",
                          power_transform_target=power_transform_target)
        data3 = read_data(target=target, file_name="BalochistanData_new.xlsx",
                          power_transform_target=power_transform_target)
    else:
        data2 = read_data(target=target, file_name="KarachiData_old.xlsx",
                          power_transform_target=power_transform_target)
        data3 = read_data(target=target, file_name="BalochistanData_old.xlsx",
                          power_transform_target=power_transform_target)

    data= pd.concat([data1, data2, data3])

    data['pcp_mm'] = data['pcp_mm'].interpolate()

    if reindex:
        data = data.reset_index()
        # remove the "index" column which is inserted when we reset_index
        data.pop("index")

    if remove_humidity:
        data.pop("rel_hum")

    return data

def tetx_data():
    data = make_whole_data("tetx_coppml")

    input_features = data.columns.tolist()[0:-1]
    output_features = data.columns.tolist()[-1:]

    dataset = DataSet(data, train_fraction=1.0, val_fraction=0.0)
    x, y = dataset.training_data()
    x = np.delete(x, [16], axis=0)
    y = np.delete(y, [16]).reshape(-1, 1)

    x = np.delete(x, [151], axis=0)
    y = np.delete(y, [151]).reshape(-1, 1)

    x = np.delete(x, [167], axis=0)
    y = np.delete(y, [167]).reshape(-1, 1)

    x = np.delete(x, [10], axis=0)
    y = np.delete(y, [10]).reshape(-1, 1)

    return x, y, input_features, output_features

x, y, input_features, output_features = tetx_data()

def get_fitted_model(ModelClass):

    model = ModelClass(model=   {
            "XGBRegressor": {
                "n_estimators": 91,
                "learning_rate": 0.3889111111111111,
                "booster": "dart",
                "random_state": 313
            }
        },


    x_transformation= [
            {
                "method": "center",
                "features": [
                    "wind_speed_mps"
                ]
            },
            {
                "method": "log2",
                "features": [
                    "wat_temp_c"
                ],
                "treat_negatives": True,
                "replace_zeros": True
            },
            {
                "method": "pareto",
                "features": [
                    "tide_cm"
                ]
            },
            {
                "method": "log10",
                "features": [
                    "sal_psu"
                ],
                "treat_negatives": True,
                "replace_zeros": True
            },
            {
                "method": "center",
                "features": [
                    "pcp_mm"
                ]
            },
            {
                "method": "quantile",
                "features": [
                    "air_p_hpa"
                ],
                "n_quantiles": 40
            }
        ],
    y_transformation=  [
            {
                "method": "log10",
                "features": [
                    "tetx_coppml"
                ],
                "treat_negatives": True,
                "replace_zeros": True
            }
        ],

                seed=313,
                output_features=output_features,
                input_features=input_features,
                split_random = False,
                cross_validator= {"TimeSeriesSplit": {"n_splits": 10}},
                verbosity=0,
                )

    # %%

    _ = model.fit(x=x, y=y)

    return model


def sobol_plots(si, show=False, path:str=None):
    total, first, second = si.to_df()

    plt.close('all')
    bar_plot(total)
    if path:
        plt.savefig(os.path.join(path, "total"), bbox_inches="tight")
    if show:
        plt.show()

    plt.close('all')
    bar_plot(first)
    if path:
        plt.savefig(os.path.join(path, "first_order"), bbox_inches="tight")
    if show:
        plt.show()

    fig, ax = plt.subplots(figsize=(16, 6))
    bar_plot(second, ax=ax)
    if path:
        plt.savefig(os.path.join(path, "first_order"), bbox_inches="tight")
    if show:
        plt.show()

    return


def bar_plot(sis_df:pd.DataFrame, sort=True, conf_col = "_conf", **kwargs):

    conf_cols = sis_df.columns.str.contains(conf_col)

    sis = sis_df.loc[:, ~conf_cols].values
    confs = sis_df.loc[:, conf_cols].values
    names = sis_df.index

    if isinstance(names[0], tuple):
        names = np.array([str(i) for i in names])

    if len(sis) == sis.size:
        confs = confs.reshape(-1, )
        sis = sis.reshape(-1,)
    else:
        raise ValueError

    if sort:
        sort_idx = np.argsort(sis)
        confs = confs[sort_idx]
        sis = sis[sort_idx]
        names = names[sort_idx]

    label = sis_df.columns[~conf_cols][0]

    ax = ep.bar_chart(sis, names, orient="v", sort=sort, rotation=90, show=False,
                   label=label, **kwargs)
    if sort:
        ax.legend(loc="upper left")
    else:
        ax.legend(loc="best")

    ax.errorbar(np.arange(len(sis)), sis, yerr=confs, fmt=".", color="black")
    return ax



def sensitivity_plots(analyzer, si, path=None, show=False):

    if analyzer == "morris":
        morris_plots(si, path=path, show=show)

    elif analyzer in ["sobol"]:
        sobol_plots(si, show, path)

    elif analyzer == "hdmr":

        plt.close('all')
        plot(si)
        if path:
            plt.savefig(os.path.join(path, "hdmr"), bbox_inches="tight")

    elif analyzer in ["pawn"]:
        plt.close('all')
        si_df = si.to_df()
        bar_plot(si_df[["CV", "median"]], conf_col="median")
        if path:
            plt.savefig(os.path.join(path, "pawn_cv"), bbox_inches="tight")
        if show:
            plt.show()

    elif analyzer == "fast":
        plt.close('all')
        si_df = si.to_df()
        bar_plot(si_df[["S1", "S1_conf"]])
        if path:
            plt.savefig(os.path.join(path, "fast_s1"), bbox_inches="tight")
        if show:
            plt.show()

        plt.close('all')
        bar_plot(si_df[["ST", "ST_conf"]])
        if path:
            plt.savefig(os.path.join(path, "fast_s1"), bbox_inches="tight")
        if show:
            plt.show()


    elif analyzer == "delta":
        plt.close('all')
        si_df = si.to_df()
        bar_plot(si_df[["delta", "delta_conf"]])
        if path:
            plt.savefig(os.path.join(path, "fast_s1"), bbox_inches="tight")
        if show:
            plt.show()

        plt.close('all')
        bar_plot(si_df[["S1", "S1_conf"]])
        if path:
            plt.savefig(os.path.join(path, "fast_s1"), bbox_inches="tight")
        if show:
            plt.show()

    elif analyzer == "rbd_fast":
        plt.close('all')
        si_df = si.to_df()
        bar_plot(si_df[["S1", "S1_conf"]])
        if path:
            plt.savefig(os.path.join(path, "rbd_fast_s1"), bbox_inches="tight")
        if show:
            plt.show()
    return


class Model(_Model):

    def sensitivity_analysis(
            self,
            data=None,
            bounds=None,
            sampler="morris",
            analyzer: Union[str, list] = "sobol",
            sampler_kwds: dict = None,
            analyzer_kwds: dict = None,
            save_plots: bool = True,
            names: List[str] = None
    ) -> dict:
        """performs sensitivity analysis of the model w.r.t input features in data.
        The model and its hyperprameters remain fixed while the input data is changed.
        Parameters
        ----------
        data :
            data which will be used to get the bounds/limits of input features. If given,
            it must be 2d numpy array. It should be remembered that the given data
            is not used during sensitivity analysis. But new synthetic data is prepared
            on which sensitivity analysis is performed.
        bounds : list,
            alternative to data
        sampler : str, optional
            any sampler_ from SALib library. For example ``morris``, ``fast_sampler``,
            ``ff``, ``finite_diff``, ``latin``, ``saltelli``, ``sobol_sequence``
        analyzer : str, optional
            any analyzer_ from SALib lirary. For example ``sobol``, ``dgsm``, ``fast``
            ``ff``, ``hdmr``, ``morris``, ``pawn``, ``rbd_fast``. You can also choose
            more than one analyzer. This is useful when you want to compare results
            of more than one analyzers. It should be noted that having more than
            one analyzers does not increases computation time except for ``hdmr``
            and ``delta`` analyzers. The ``hdmr`` and ``delta`` analyzers ane computation
            heavy. For example
            >>> analyzer = ["morris", "sobol", "rbd_fast"]
        sampler_kwds : dict
            keyword arguments for sampler
        analyzer_kwds : dict
            keyword arguments for analyzer
        save_plots : bool, optional
        names : list, optional
            names of input features. If not given, names of input features will be used.
        Returns
        -------
        dict :
            a dictionary whose keys are names of analyzers and values and sensitivity
            results for that analyzer.
        Examples
        --------
        >>> from ai4water import Model
        >>> from ai4water.datasets import busan_beach
        >>> df = busan_beach()
        >>> input_features=df.columns.tolist()[0:-1]
        >>> output_features = df.columns.tolist()[-1:]
        ... # build the model
        >>> model=Model(model="RandomForestRegressor",
        >>>     input_features=input_features,
        >>>     output_features=output_features)
        ... # train the model
        >>> model.fit(data=df)
        .. # perform sensitivity analysis
        >>> si = model.sensitivity_analysis(data=df[input_features].values,
        >>>                    sampler="morris", analyzer=["morris", "sobol"],
        >>>                        sampler_kwds={'N': 100})
        .. _sampler:
            https://salib.readthedocs.io/en/latest/api/SALib.sample.html
        .. _analyzer:
            https://salib.readthedocs.io/en/latest/api/SALib.analyze.html
        """
        try:
            import SALib
        except (ImportError, ModuleNotFoundError):
            warnings.warn("""
            You must have SALib library installed in order to perform sensitivity analysis.
            Please install it using 'pip install SALib' and make sure that it is importable
            """)
            return {}

        #from ai4water.postprocessing._sa import sensitivity_analysis, sensitivity_plots
        #from ai4water.postprocessing._sa import _make_predict_func

        if data is not None:
            if not isinstance(data, np.ndarray):
                assert isinstance(data, pd.DataFrame)
                data = data.values
            x = data

            # calculate bounds
            assert isinstance(x, np.ndarray)
            bounds = []
            for feat in range(x.shape[1]):
                bound = [np.min(x[:, feat]), np.max(x[:, feat])]
                bounds.append(bound)
        else:
            assert bounds is not None
            assert isinstance(bounds, list)
            assert all([isinstance(bound, list) for bound in bounds])

        analyzer_kwds = analyzer_kwds or {}

        if self.lookback > 1:
            if self.category == "DL":
                func = _make_predict_func(self, verbose=0)
            else:
                func = _make_predict_func(self)
        else:
            func = self.predict

        results = sensitivity_analysis(
            sampler,
            analyzer,
            func,
            bounds=bounds,
            sampler_kwds=sampler_kwds,
            analyzer_kwds=analyzer_kwds,
            names=names or self.input_features
        )

        if save_plots:
            for _analyzer, result in results.items():
                res_df = result.to_df()
                if isinstance(res_df, list):
                    for idx, res in enumerate(res_df):
                        fname = os.path.join(self.path, f"{_analyzer}_{idx}_results.csv")
                        res.to_csv(fname)
                else:
                    res_df.to_csv(os.path.join(self.path, f"{_analyzer}_results.csv"))

                sensitivity_plots(_analyzer, result, self.path)

        return results


def sensitivity_analysis(
        sampler: str,
        analyzer: Union[str, list],
        func: Callable,
        bounds: list,
        sampler_kwds: dict = None,
        analyzer_kwds: dict = None,
        names: list = None,
        **kwargs
) -> dict:
    """
    Parameters
    ----------
    sampler :
    analyzer :
    func :
    bounds :
    sampler_kwds :
    analyzer_kwds :
    names :
    **kwargs :
    """
    sampler = importlib.import_module(f"SALib.sample.{sampler}")

    if names is None:
        names = [f"Feat{i}" for i in range(len(bounds))]

    # Define the model inputs
    problem = {
        'num_vars': len(bounds),
        'names': names,
        'bounds': bounds
    }

    sampler_kwds = sampler_kwds or {'N': 100}

    param_values = sampler.sample(problem=problem, **sampler_kwds)
    print("total samples:", len(param_values))

    y = func(x=param_values, **kwargs)

    y = np.array(y)

    assert np.size(y) == len(y), f"output must be 1 dimensional"
    y = y.reshape(-1, )

    results = {}
    if isinstance(analyzer, list):
        for _analyzer in analyzer:
            print(f"Analyzing with {_analyzer}")
            results[_analyzer] = analyze(_analyzer, param_values, y, problem, analyzer_kwds)
    else:
        assert isinstance(analyzer, str)
        results[analyzer] = analyze(analyzer, param_values, y, problem, analyzer_kwds)

    return results


def analyze(analyzer, param_values, y, problem, analyzer_kwds):
    _analyzer = importlib.import_module(f"SALib.analyze.{analyzer}")
    analyzer_kwds = analyzer_kwds or {}

    if analyzer in ["hdmr",
                    "morris",
                    "dgsm",
                    "ff",
                    "pawn",
                    "rbd_fast", "delta",
                    ] and 'X' not in analyzer_kwds:
        analyzer_kwds['X'] = param_values

    Si = _analyzer.analyze(problem=problem, Y=y, **analyzer_kwds)

    if 'X' in analyzer_kwds:
        analyzer_kwds.pop('X')

    return Si


def _make_predict_func(model, **kwargs):

    from ai4water.preprocessing import DataSet

    lookback = model.config["ts_args"]['lookback']

    def func(x):
        x = pd.DataFrame(x, columns=model.input_features)

        ds = DataSet(data=x,
                     ts_args=model.config["ts_args"],
                     input_features=model.input_features,
                     train_fraction=1.0,
                     val_fraction=0.0,
                     verbosity=0)

        x, _ = ds.training_data()
        p = model.predict(x=x, **kwargs)

        return np.concatenate([p, np.zeros((lookback-1, 1))])

    return func


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


class PartialDependencePlot1(PartialDependencePlot):

    def __init__(
            self,
            model: Callable,
            data,
            feature_names=None,
            num_points: int = 100,
            path=None,
            save: bool = True,
            show: bool = True,
            **kwargs
    ):
        """Initiates the class
        Parameters
        ----------
            model : Callable
                the trained/calibrated model which must be callable. It must take the
                `data` as input and sprout an array of predicted values. For example
                if you are using Keras/sklearn model, then you must pass model.predict
            data : np.ndarray, pd.DataFrame
                The inputs to the `model`. It can numpy array or pandas DataFrame.
            feature_names : list, optional
                Names of features. Used for labeling.
            num_points : int, optional
                determines the grid for evaluation of `model`
            path : str, optional
                path to save the plots. By default the results are saved in current directory
            show:
                whether to show the plot or not
            save:
                whether to save the plot or not
            **kwargs :
                any additional keyword arguments for `model`
        """

        self.model = model
        self.num_points = num_points
        self.xmin = "percentile(0)"
        self.xmax = "percentile(100)"
        self.kwargs = kwargs

        if isinstance(data, pd.DataFrame):
            if feature_names is None:
                feature_names = data.columns.tolist()
            data = data.values

        if not os.path.exists(path):
            os.makedirs(path)

        self.path = path
        self.data = data
        self.features = feature_names
        self.save = save
        self.show = show

    @property
    def data_is_2d(self):
        if isinstance(self.data, np.ndarray) and self.data.ndim == 2:
            return True
        elif isinstance(self.data, pd.DataFrame):
            return True
        else:
            return False

    @property
    def data_is_3d(self):
        if isinstance(self.data, np.ndarray) and self.data.ndim == 3:
            return True
        return False

    @property
    def single_source(self):
        if isinstance(self.data, list) and len(self.data) > 1:
            return False
        else:
            return True

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, features):
        if self.data_is_2d:
            if type(self.data) == pd.DataFrame:

                features = self.data.columns.to_list()
            elif features is None:
                features = [f"Feature {i}" for i in range(self.data.shape[-1])]
            else:
                assert isinstance(features, list) and len(features) == self.data.shape[-1], f"""
                    features must be given as list of length {self.data.shape[-1]} 
                    but are of len {len(features)}
                    """

                features = features
        elif not self.single_source and features is None:
            features = []
            for data in self.data:
                if isinstance(data, pd.DataFrame):
                    _features = data.columns.to_list()
                else:
                    _features = [f"Feature {i}" for i in range(data.shape[-1])]

                features.append(_features)

        elif self.data_is_3d and features is None:
            features = [f"Feature {i}" for i in range(self.data.shape[-1])]

        self._features = features

    def plot_1d(
            self,
            feature,
            show_dist: bool = True,
            show_dist_as: str = "hist",
            ice: bool = True,
            feature_expected_value: bool = False,
            model_expected_value: bool = False,
            show_ci: bool = False,
            show_minima: bool = False,
            ice_only: bool = False,
            ice_color: str = "lightblue",
    ):
        """partial dependence plot in one dimension
        Parameters
        ----------
            feature :
                the feature name for which to plot the partial dependence
            show_dist :
                whether to show actual distribution of data or not
            show_dist_as :
                one of "hist" or "grid"
            ice :
                whether to show individual component elements on plot or not
            feature_expected_value :
                whether to show the average value of feature on the plot or not
            model_expected_value :
                whether to show average prediction on plot or not
            show_ci :
                whether to show confidence interval of pdp or not
            show_minima :
                whether to indicate the minima or not
            ice_only : bool, False
                whether to show only ice plots
            ice_color :
                color for ice lines. It can also be a valid maplotlib
                `colormap <https://matplotlib.org/3.5.1/tutorials/colors/colormaps.html>`_
        """
        if isinstance(feature, list) or isinstance(feature, tuple):
            raise NotImplementedError
        else:
            if self.single_source:
                if self.data_is_2d:
                    ax = self._plot_pdp_1dim(
                        *self._pdp_for_2d(self.data, feature),
                        self.data, feature,
                        show_dist=show_dist,
                        show_dist_as=show_dist_as,
                        ice=ice,
                        feature_expected_value=feature_expected_value,
                        show_ci=show_ci, show_minima=show_minima,
                        model_expected_value=model_expected_value,
                        show=self.show,
                        save=self.save,
                        ice_only=ice_only,
                        ice_color=ice_color)
                elif self.data_is_3d:
                    for lb in range(self.data.shape[1]):
                        ax = self._plot_pdp_1dim(
                            *self._pdp_for_2d(self.data, feature, lb),
                            data=self.data,
                            feature=feature,
                            lookback=lb,
                            show_ci=show_ci,
                            show_minima=show_minima,
                            show_dist=show_dist,
                            show_dist_as=show_dist_as,
                            ice=ice,
                            feature_expected_value=feature_expected_value,
                            model_expected_value=model_expected_value,
                            show=self.show,
                            save=self.save,
                            ice_only=ice_only,
                            ice_color=ice_color)
                else:
                    raise ValueError(f"invalid data shape {self.data.shape}")
            else:
                for data in self.data:
                    if self.data_is_2d:
                        ax = self._pdp_for_2d(data, feature)
                    else:
                        for lb in []:
                            ax = self._pdp_for_2d(data, feature, lb)
        return ax

    def _plot_pdp_1dim(
            self,
            pd_vals, ice_vals, data, feature,
            lookback=None,
            show_dist=True, show_dist_as="hist",
            ice=True, show_ci=False,
            show_minima=False,
            feature_expected_value=False,
            model_expected_value=False,
            show=True, save=False, ax=None,
            ice_color="lightblue",
            ice_only=False,
    ):

        xmin, xmax = compute_bounds(self.xmin,
                                    self.xmax,
                                    self.xv(data, feature, lookback))

        if ax is None:
            fig = plt.figure()
            ax = fig.add_axes((0.1, 0.3, 0.8, 0.6))

        xs = self.grid(data, feature, lookback)

        ylabel = "E[f(x) | " + feature + "]"
        if ice:
            n = ice_vals.shape[1]
            if ice_color in plt.colormaps():
                colors = plt.get_cmap(ice_color)(np.linspace(0, 0.8, n))
            else:
                colors = [ice_color for _ in range(n)]

            ice_linewidth = min(1, 50 / n)  # pylint: disable=unsubscriptable-object
            for _ice in range(n):
                ax.plot(xs, ice_vals[:, _ice], color=colors[_ice],
                        linewidth=ice_linewidth, alpha=1)
            ylabel = "f(x) | " + feature

        if show_ci:
            std = np.std(ice_vals, axis=1)
            upper = pd_vals + std
            lower = pd_vals - std
            color = '#66C2D7'
            if ice_color != "lightblue":
                if ice_color not in plt.colormaps():
                    color = ice_color

            ax.fill_between(xs, upper, lower, alpha=0.14, color=color)

        # the line plot
        if not ice_only:
            ax.plot(xs, pd_vals, color='blue', linewidth=2, alpha=1)

        title = None
        if lookback is not None:
            title = f"lookback: {lookback}"
        process_axis(ax,
                     ylabel=ylabel,
                     ylabel_kws=dict(fontsize=20),
                     right_spine=False,
                     top_spine=False,
                     tick_params=dict(labelsize=11),
                     xlabel=feature,
                     xlabel_kws=dict(fontsize=20),
                     title=title)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        ax2 = ax.twinx()

        if show_dist:
            xv = self.xv(data, feature, lookback)
            if show_dist_as == "hist":

                ax2.hist(xv, 50, density=False, facecolor='black', alpha=0.1,
                         range=(xmin, xmax))
            else:
                _add_dist_as_grid(fig, xv, other_axes=ax, xlabel=feature,
                                  xlabel_kws=dict(fontsize=20))

        process_axis(ax2,
                     right_spine=False,
                     top_spine=False,
                     left_spine=False,
                     bottom_spine=False,
                     ylim=(0, data.shape[0]))
        ax2.xaxis.set_ticks_position('bottom')
        ax2.yaxis.set_ticks_position('left')
        ax2.yaxis.set_ticks([])

        if feature_expected_value:
            self._add_feature_exp_val(ax2, ax, xmin, xmax, data, feature, lookback)

        if model_expected_value:
            self._add_model_exp_val(ax2, ax, data)

        if show_minima:
            minina = self.model(data, **self.kwargs).min()
            ax.axvline(minina, linestyle="--", color="r", lw=1)

        if save:
            lookback = lookback or ''
            fname = os.path.join(self.path, f"pdp_{feature}_{lookback}")
            plt.savefig(fname, bbox_inches="tight", dpi=400)

        if show:
            plt.show()

        return ax

    def _pdp_for_2d(self, data, feature, lookback=None):
        ind = self._feature_to_ind(feature)

        xs = self.grid(data, feature, lookback)

        data_temp = data.copy()

        # instead of calling the model for each num_point, prepare the data
        # stack it in 'data_all' and call the model only once
        total_samples = len(data) * self.num_points
        data_all = np.full((total_samples, *data.shape[1:]), np.nan)

        pd_vals = np.full(self.num_points, np.nan)
        ice_vals = np.full((self.num_points, data.shape[0]), np.nan)

        st, en = 0, len(data)
        for i in range(self.num_points):

            if data.ndim == 3:
                data_temp[:, lookback, ind] = xs[i]
            else:
                data_temp[:, ind] = xs[i]
                data_all[st:en] = data_temp

            st = en
            en += len(data)

        predictions = self.model(data_all, **self.kwargs)

        st, en = 0, len(data)
        for i in range(self.num_points):
            pred = predictions[st:en]
            pd_vals[i] = pred.mean()
            ice_vals[i, :] = pred.reshape(-1, )
            st = en
            en += len(data)

        return pd_vals, ice_vals


def plot_convergence(
        results: dict,
        method: str,
        item: str, sub_method: str = '',
        xlabel_kws = None,
        ylabel_kws = None,
        xticklabel_kws = None,
        yticklabel_kws = None,
        leg_kws = None,
        labels=None,
        figsize=(14, 8)
):
    random.seed(313)

    _n = list(results.keys())[0]
    meth = list(results[_n].keys())[0]
    names = results[_n][meth]["names"]

    markers = ["--o", "--*", "--.", "--^"]

    convergence = {n: [] for n in names}

    for n, result in results.items():
        method_si = result[method]
        method_si_df = method_si.to_df()

        if method == "sobol":
            total, first, second = method_si_df
            if sub_method == "first":
                method_si_df = first
            elif sub_method == "second":
                method_si_df = second
            else:
                method_si_df = total

        for feature in convergence.keys():
            val = method_si_df.loc[feature, item]

            convergence[feature].append(val)

    fig, ax = plt.subplots(figsize=figsize)

    for idx, (key, val) in enumerate(convergence.items()):
        marker = random.choice(markers)
        if labels is None:
            label = key
        else:
            label = labels[idx]
        ax = ep.plot(val, marker, label=label, show=False, ax=ax)

    leg_kws = leg_kws or {"fontsize": 14}
    ax.legend(loc=(1.01, 0.01), **leg_kws)

    ylabel_kws = ylabel_kws or {'fontsize': 14}
    ax.set_ylabel(item, **ylabel_kws)
    xlabel_kws = xlabel_kws or {'fontsize':14}
    ax.set_xlabel("Number of Model Evaluations", **xlabel_kws)
    ax.set_title(f"Convergence of {method} Sensitivity Analysis {sub_method}", fontsize=14)

    xticklabels = list(results.keys())
    ax.set_xticks(np.arange(len(xticklabels)))
    xticklabel_kws = xticklabel_kws or {'fontsize': 12}
    ax.set_xticklabels(xticklabels, **xticklabel_kws)

    #yticklabel_kws = yticklabel_kws or {'fontsize': 12}
    #ax.set_yticklabels(ax.get_yticklabels(), **yticklabel_kws)

    return ax


def confidenc_interval(model, X_train, y_train, X_test, alpha,
                    n_splits=5):
    def generate_results_dataset(preds, ci):
        df = pd.DataFrame()
        df['prediction'] = preds
        if ci >= 0:
            df['upper'] = preds + ci
            df['lower'] = preds - ci
        else:
            df['upper'] = preds - ci
            df['lower'] = preds + ci

        return df

    model.fit(X_train, y_train)
    residuals = y_train - model.predict(X_train)
    ci = np.quantile(residuals, 1 - alpha)
    preds = model.predict(X_test)
    df = generate_results_dataset(preds.reshape(-1, ), ci)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    res = []
    estimators = []
    for train_index, test_index in kf.split(X_train):
        X_train_, X_test_ = X_train[train_index], X_train[test_index]
        y_train_, y_test_ = y_train[train_index], y_train[test_index]

        model.fit(X_train_, y_train_)
        estimators.append(model)
        _pred = model.predict(X_test_)
        res.extend(list(y_test_ - _pred.reshape(-1, )))

    y_pred_multi = np.column_stack([e.predict(X_test) for e in estimators])

    ci = np.quantile(res, 1 - alpha)
    top = []
    bottom = []
    for i in range(y_pred_multi.shape[0]):
        if ci > 0:
            top.append(np.quantile(y_pred_multi[i] + ci, 1 - alpha))
            bottom.append(np.quantile(y_pred_multi[i] - ci, 1 - alpha))
        else:
            top.append(np.quantile(y_pred_multi[i] - ci, 1 - alpha))
            bottom.append(np.quantile(y_pred_multi[i] + ci, 1 - alpha))

    preds = np.median(y_pred_multi, axis=1)
    df = pd.DataFrame()
    df['pred'] = preds
    df['upper'] = top
    df['lower'] = bottom

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.fill_between(np.arange(len(df)), df['upper'], df['lower'], alpha=0.5, color='C1')
    p1 = ax.plot(df['pred'], color="C1", label="Prediction")
    p2 = ax.fill(np.NaN, np.NaN, color="C1", alpha=0.5)
    percent = int((1 - alpha) * 100)
    ax.legend([(p2[0], p1[0]), ], [f'{percent}% Confidence Interval'],
              fontsize=12)
    ax.set_xlabel("Test Samples", fontsize=12)
    target = model.output_features[0]
    ax.set_ylabel(target, fontsize=12)
    fpath = os.path.join(model.path, f"{percent}_interval_{target}")
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.show()

    return

