from . import utils
import os
import numpy as np


esol = utils.from_csv(os.path.dirname(utils.__file__) + "/esol.csv")
freesolv = utils.from_csv(
    os.path.dirname(utils.__file__) + "/SAMPL.csv", smiles_col=1, y_cols=[2]
)
lipophilicity = utils.from_csv(
    os.path.dirname(utils.__file__) + "/Lipophilicity.csv"
)
moonshot = utils.from_csv(
    os.path.dirname(utils.__file__) + "/moonshot.csv",
    smiles_col=0,
    y_cols=[6],
    scale=0.01,
    dropna=True,
)
