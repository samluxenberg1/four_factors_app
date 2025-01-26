import numpy as np
import pandas as pd
import statsmodels.api as sm


from src import data_processing


import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st


#data_path = os.path.join(root_dir,"data", "four_factors.csv")
#df = pd.read_csv(data_path)
df = st.session_state["data"]
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
st.write(rf"$$\hat{{Win \%}} = {model_ff.params[0]:.2f} + {model_ff.params[1]:.2f}\times \text{{EFG\_PCT\_d}} + {model_ff.params[2]:.2f}\times \text{{FTA\_Rate\_d}}+{model_ff.params[3]:.2f}\times \text{{TM\_TOV\_PCT\_d}}+{model_ff.params[4]:.2f}\times \text{{OREB\_PCT\_d}}$$")

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

