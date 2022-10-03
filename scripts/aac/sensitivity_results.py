"""
=========================
aac sensitivity
=========================
"""


import os
import pandas as pd
import matplotlib.pyplot as plt

from aac_utils import get_fitted_model, sobol_plots, sensitivity_plots, plot_convergence

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

import SALib
from ai4water.postprocessing._sa import morris_plots
print(SALib.__version__)

#%%

res = model.sensitivity_analysis(data=train_x.values,
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

res = model.sensitivity_analysis(data=train_x.values,
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
        data=train_x.values,
        sampler="morris",
        analyzer=["morris", "sobol", "pawn", "rbd_fast"],
        save_plots=False,
        sampler_kwds = {"N": n},
        analyzer_kwds = {'print_to_console': False},
        names = train_x.columns.tolist()
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