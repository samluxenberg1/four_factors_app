import os
from four_factors_app import definitions
import statsmodels.api as sm
import pandas as pd

def processing(df: pd.DataFrame):
    four_off_factors = ['EFG_PCT', 'FTA_RATE', 'TM_TOV_PCT', 'OREB_PCT']
    four_def_factors = ['OPP_EFG_PCT', 'OPP_FTA_RATE', 'OPP_TOV_PCT', 'OPP_OREB_PCT']

    X = df[four_off_factors + four_def_factors]

    y = df['W']
    y2 = df['W_PCT']

    X_diff = X.copy()
    X_diff['EFG_PCT_d'] = X_diff['EFG_PCT'] - X_diff['OPP_EFG_PCT']
    X_diff['FTA_RATE_d'] = X_diff['FTA_RATE'] - X_diff['OPP_FTA_RATE']
    X_diff['TM_TOV_PCT_d'] = X_diff['TM_TOV_PCT'] - X_diff['OPP_TOV_PCT']
    X_diff['OREB_PCT_d'] = X_diff['OREB_PCT'] - X_diff['OPP_OREB_PCT']
    X_diff = X_diff.drop(
        ['EFG_PCT', 'OPP_EFG_PCT', 'FTA_RATE', 'OPP_FTA_RATE','TM_TOV_PCT', 'OPP_TOV_PCT', 'OREB_PCT', 'OPP_OREB_PCT'],
                         axis=1
    )

    X1_diff = sm.add_constant(X_diff)

    return X1_diff, y, y2



