import os
import sys
from pathlib import Path

# Add root directory to sys.path before importing your modules
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
print(root_dir)

#from four_factors_app import definitions
from src import data_processing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro

import streamlit as st

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#root_dir = Path(__file__).resolve().parent.parent
#sys.path.append(str(root_dir))
#print(sys.path)

if __name__ == "__main__":
    data_path = os.path.join(root_dir,"data", "four_factors.csv")
    df = pd.read_csv(data_path)

    # Process Data
    X1_diff, y, y2 = data_processing.processing(df=df)
    df = pd.concat([df, X1_diff.drop('const', axis=1)], axis=1)

    teams = ['All'] + list(df['TEAM_NAME'].unique())
    seasons = ['All'] + list(df['season'].unique())
    min_season = int(df['season'].min())
    max_season = int(df['season'].max())

    selected_teams = st.sidebar.multiselect("Select Teams", options=teams, default='All')
    selected_seasons = st.sidebar.slider("Select Seasons", min_value=min_season, max_value=max_season, value=(min_season, max_season))

    if 'All' in selected_teams:
        filtered_teams = df['TEAM_NAME'].unique()
    else:
        filtered_teams = selected_teams

    #if 'All' in selected_seasons:
    #    filtered_seasons = df['season'].unique()
    #else:
    #    filtered_seasons = selected_seasons

    filtered_df = df[(df['TEAM_NAME'].isin(filtered_teams)) & (df['season'].between(selected_seasons[0],selected_seasons[1]))]
    st.write(f"# Four Factor Model")
    st.write("## A Retrospective Analysis")

    st.write(f"## Data")
    st.dataframe(filtered_df.drop('TEAM_ID', axis=1))
    st.write(f"Displaying {len(filtered_df)} records")

    # EDA Plots
    st.write(f"## EDA Plots")
    feature_columns = X1_diff.drop('const', axis=1).columns.tolist()
    orig_factors = ['EFG_PCT','FTA_RATE','TM_TOV_PCT','OREB_PCT']
    fig0, ax0 = plt.subplots(2,2, figsize=(14,14))
    ax0 = ax0.ravel()

    for idx, feature in enumerate(orig_factors):
        ax = ax0[idx]

        # Compute League Average by Season Using Filtered data
        league_avg = df.groupby('season')[feature].agg('mean').reset_index()
        filtered_league_avg = league_avg[league_avg['season'].between(selected_seasons[0], selected_seasons[1])]

        # Plot scatter plot for each selected team in filtered data
        for team in filtered_df['TEAM_NAME'].unique():
            team_df = filtered_df[filtered_df['TEAM_NAME'] == team]
            if filtered_df['TEAM_NAME'].nunique() > 4:
                sns.scatterplot(x=team_df['season'], y=team_df[feature],ax=ax, color='gray', alpha=.5)
            else:
                sns.scatterplot(x=team_df['season'], y=team_df[feature], ax=ax, label=f"{team}", alpha=.5)

        # Plot line for league average
        sns.lineplot(x=filtered_league_avg['season'].unique(), y=filtered_league_avg[feature], ax=ax, label='League Avg', color='red')

        ax.set_title(f"{feature} Over Seasons")
        ax.set_xlabel("Season")
        ax.set_xticks(filtered_league_avg['season'])
        ax.set_xticklabels(filtered_league_avg['season'].astype(int), rotation=45)

        ax.set_ylabel(feature)
        ax.legend(loc='best')

    st.pyplot(fig0)

    # Correlation Heatmap
    df_corr = filtered_df[orig_factors + ['W_PCT']].corr()
    fig1, ax1 = plt.subplots()
    sns.heatmap(
        df_corr, annot=True,cmap='coolwarm',
        vmin=-1,vmax=1,fmt='.2f',
        square=True,
        cbar_kws={'label':'Correlation Coefficient'},
        ax=ax1
    )
    ax1.set_title('Correlation Matrix')
    st.pyplot(fig1)

    # Linear Regression Output

    # Fit Linear Model
    model_ff = sm.OLS(y2, X1_diff).fit()

    # Training Predictions
    df['y_pred'] = model_ff.predict()
    df['res'] = df['W_PCT'] - df['y_pred']

    # Filter
    filtered_preds = df.loc[filtered_df.index, 'y_pred']
    filtered_res = df.loc[filtered_df.index, 'res']

    st.write("## Model Output")
    st.write(f"### Estimated Model")
    st.write(
        rf"$$\hat{{y}} = {model_ff.params[0]:.2f} + {model_ff.params[1]:.2f}x_1 + {model_ff.params[2]:.2f}x_2+{model_ff.params[3]:.2f}x_3+{model_ff.params[4]:.2f}x_4$$"
    )
    st.write(rf"$$\hat{{y}} = {model_ff.params[0]:.2f} + {model_ff.params[1]:.2f}\times \text{{EFG\_PCT\_d}} + {model_ff.params[2]:.2f}\times \text{{FTA\_Rate\_d}}+{model_ff.params[3]:.2f}\times \text{{TM\_TOV\_PCT\_d}}+{model_ff.params[4]:.2f}\times \text{{OREB\_PCT\_d}}$$")
    st.write(r"$x_1$ = Difference in eFG\%")
    st.write(r"$x_2$ = Difference in FTA Rate")
    st.write(r"$x_3$ = Difference in TOV\%")
    st.write(r"$x_4$ = Difference in OREB\%")
    # Plots
    st.write("### Residuals vs. Predicted Values")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=filtered_preds, y=filtered_res, alpha=.5, ax=ax2)
    ax2.axhline(y=0, color='red', linestyle='dashed')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    st.pyplot(fig2)

    st.write("### Residuals vs. Features")
    feature_columns = X1_diff.drop('const', axis=1).columns.tolist()
    fig3, ax3 = plt.subplots(2,2, figsize=(14,14))
    ax3 = ax3.ravel()
    for idx, feature in enumerate(feature_columns):
        ax = ax3[idx]
        sns.scatterplot(x=filtered_df[feature], y=filtered_res, alpha=.5, ax=ax)
        ax.axhline(y=0, color='red', linestyle='dashed')
        ax.set_xlabel(f'{feature}')
        ax.set_ylabel('Residuals')
        ax.set_title(f'Residuals vs. {feature}')

    st.pyplot(fig3)

    st.write("### Q-Q Plot for Residuals")
    fig4, ax4 = plt.subplots()
    sm.qqplot(filtered_res, line='q', ax=ax4)
    st.pyplot(fig4)

    st.write("## Model Metrics")
    st.write(f"R-Squared: {float(model_ff.rsquared): .3f}")
    st.write(f"Adj. R-Squared: {float(model_ff.rsquared_adj): .3f}")
    st.write(f"MSE: {float(model_ff.mse_resid): .3f}")
    rmse = float(np.sqrt(model_ff.mse_resid))
    st.write(f"RMSE: {rmse: .3f}")
    st.write(f"RMSE2: {float(np.mean(model_ff.resid**2)**.5): .3f}")

    st.write(f"On average, our model is off by {rmse*100: .2f} percentage points. This equates to {rmse*82: .2f} regular season games.")


