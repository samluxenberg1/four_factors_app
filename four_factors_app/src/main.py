import os
from four_factors_app import definitions
from four_factors_app.src import data_processing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro

import streamlit as st

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

if __name__ == "__main__":
    data_path = os.path.join(definitions.DATA_DIR, "four_factors.csv")
    df = pd.read_csv(data_path)

    # Process Data
    X1_diff, y, y2 = data_processing.processing(df=df)

    # Fit Linear Model
    model_ff = sm.OLS(y2, X1_diff).fit()
    print(model_ff.summary())

    # Training Predictions
    y_train_pred = model_ff.predict()
    res_train = y2 - y_train_pred

    st.write("My *first* app!")
    # Plot
    sns.histplot(res_train, alpha=.5)
    plt.axvline(x=0, color='red', linestyle='dashed')
    plt.xlabel('Residuals')
    plt.show()

    plt.scatter(x=y_train_pred, y=res_train, alpha=.5)
    plt.axhline(y=0, color='red', linestyle='dashed')
    plt.xlabel('Predicted Win Percentage')
    plt.ylabel('Residuals')
    st.pyplot(plt)
