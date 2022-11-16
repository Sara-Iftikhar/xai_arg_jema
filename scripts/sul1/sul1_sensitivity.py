"""
=========================
sul1 sensitivity
=========================
"""

import os
import matplotlib.pyplot as plt

from sul1_utils import get_fitted_model, sobol_plots, sensitivity_plots, Model, plot_convergence, sul1_data

# %%
model = get_fitted_model(Model)

# %%

x, _, input_features, _ = sul1_data()

import SALib
from ai4water.postprocessing._sa import morris_plots
print(SALib.__version__)

#%%


res = model.sensitivity_analysis(data=x,
                                 sampler="morris",
                                 analyzer=["sobol", "pawn", "morris", "rbd_fast"],
                                 sampler_kwds={"N": 20000}
                                )

#%%

morris_plots(res["morris"], show=True)

#%%

sobol_plots(res["sobol"])

#%%

sensitivity_plots("rbd_fast", res["rbd_fast"], show=True)

#%%

sensitivity_plots("pawn", res["pawn"], show=True)

#%%

res = model.sensitivity_analysis(data=x,
                                 sampler="fast_sampler",
                                 analyzer=["fast"],
                                 sampler_kwds={"N": 20000}
                                )

#%%
# convergence plots
# -------------------

results = {}
for n in [100, 200, 400, 800, 1600, 3200, 6400, 10_000, 20_000]:
    print(f"n is {n}")
    results[n] = model.sensitivity_analysis(
        data=x,
        sampler="morris",
        analyzer=["morris", "sobol", "pawn", "rbd_fast"],
        save_plots=False,
        sampler_kwds = {"N": n},
        analyzer_kwds = {'print_to_console': False},
        names = input_features
    )

plot_convergence(
    results, "morris", "mu_star",
    #labels=list(column_names.values()),
    leg_kws={"fontsize": 12},
    xlabel_kws={"fontsize": 12},
    ylabel_kws={"fontsize": 12},
    xticklabel_kws={"fontsize": 10},
    figsize=(8, 5)
)
plt.savefig(os.path.join(model.path, "morris_convergence.png"), bbox_inches="tight", dpi=300)
plt.tight_layout()
plt.show()

plot_convergence(results, "sobol", "ST",
                 leg_kws={"fontsize": 12},
                 xlabel_kws={"fontsize": 12},
                 ylabel_kws={"fontsize": 12},
                 xticklabel_kws={"fontsize": 10},
figsize=(8, 5)
                 )
plt.savefig(os.path.join(model.path, "sobol_convergence.png"), bbox_inches="tight", dpi=300)
plt.tight_layout()
plt.show()

plot_convergence(results, "pawn", "CV",
                 leg_kws={"fontsize": 12},
                 xlabel_kws={"fontsize": 12},
                 ylabel_kws={"fontsize": 12},
                 xticklabel_kws={"fontsize": 10},
                 figsize=(8, 5)
                 )
plt.savefig(os.path.join(model.path, "pawn_convergence.png"), bbox_inches="tight", dpi=300)
plt.tight_layout()
plt.show()


plot_convergence(results, "rbd_fast", "S1",
                 leg_kws={"fontsize": 12},
                 xlabel_kws={"fontsize": 12},
                 ylabel_kws={"fontsize": 12},
                 xticklabel_kws={"fontsize": 10},
                 figsize=(8, 5)
                 )
plt.savefig(os.path.join(model.path, "fast_convergence.png"), bbox_inches="tight", dpi=300)
plt.tight_layout()
plt.show()