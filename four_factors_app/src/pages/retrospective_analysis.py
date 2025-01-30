import numpy as np
import pandas as pd
import statsmodels.api as sm


from src import data_processing


import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st


st.write("# Retrospective Analysis")
st.write(
    """
    On this page, we perform a retrospective analysis to try to understand the relationships between the factors and 
    the target variable. One of the most critical aspects of doing data science in the real world is thinking ahead. 
    As the data scientist, we want to be able to answer the following: 
     - What is the ultimate goal of the project?
     - Who will be the end-user of the work?
     - How will the end-user use the work?
     - What will this work help the end-user to achieve?
     
     ## Scenario
     Imagine an NBA season just ended and the coach of a struggling team has asked us to identify areas of improvement,
     so that the coaching staff determine an improvement plan during the off-season. What do we currently know about 
     the team? Well, we have a complete season's worth of information, for example, season-level factor variables and 
     total wins (as well as win %). Given that we know how well the team performed overall, we can try to draw some 
     inferences at a slightly more granular level.   
     
     ## Goal
     We want to identify key variables that *explain* overall team performance, understand their impact on winning, and
     translate this understanding into actionable recommendations.
     
     ## Methodology 
     One way to accomplish this is to estimate a linear regression model, assess its validity, interpret its estimated 
     coefficients, and convert them to wins. Once we have this information in hand, we can recommend which areas the 
     coaching staff should focus on for developing off-season and training camp plans.
    """
)

st.write(
    """
     ## Decisions
     Before we can do all of that, we need to understand and be able to justify various data and modeling decisions.
     - How much data is necessary to estimate a trustworthy model? Do we need to use all seasons going back to 2002?
     - How much data is relevant? Should we include *outlier* seasons?
        - Irregular Seasons
            - The 2011-2012 season experienced a lockout. As a result, the regular season only lasted for 66 games. 
            - In the 2012-2013 season, a game between the Boston Celtics and the Indiana Pacers was cancelled and never 
        rescheduled due to the Boston Marathon Bombing. As a result, the Boston Celtics and the Indiana Pacers only 
        played 81 regular season games compared to the rest of the league which played 82. 
            - In the 2019-2020 season, the league shut down for a number of months due to the COVID-19 pandemic, only to 
        resume the regular season in Disney World. Teams played varying numbers of regular season games ranging from 
        64 to 75 games played. 
            - The COVID-19 pandemic also affected the 2020-2021 season forcing teams to all play 72 regular season games.
        - Some Significant NBA Rule Changes beginning in the 2001-2002 Season
            - 2001-2002: Elimination of Illegal Defense, Introduction of Defensive Three-Second Violation, Reduction
            in Time Allowed to Advance Ball Past Mid-Court
            - 2004-2005: Elimination of Hand Checking
            - 2017-2018: Introduction of the 'Harden Rule' to limit players from getting shooting fouls from cheap contact, 
            Introduction of the 'Zaza Rule' to eliminate reckless closeouts against shooters
    - What should our target variable be? 
        
    When making such decisions, think both from a statistical and a stakeholder perspective. What are the most 
    statistically sound answers and what is most important to the stakeholder? Would one decision make the stakeholder
    question our recommendations more than another? 
    """
)

selected_target_name = st.selectbox(
    "Choose a target variable",
    ("Wins", "Win %"),
    index=1,
    placeholder="Select target variable..."
)
if selected_target_name == "Wins":
    selected_target = "W"
else:
    selected_target = "W_PCT"

df = st.session_state["data"]
# Process Data
X1_diff, y, y2 = data_processing.processing(df=df)
df = pd.concat([df, X1_diff.drop('const', axis=1)], axis=1)
st.dataframe(df.head())
seasons = ['All'] + list(df['season'].unique())
min_season = int(df['season'].min())
max_season = int(df['season'].max())
selected_seasons = st.slider(
    "Choose a range of seasons",
    min_value=min_season, max_value=max_season,
    value=(min_season, max_season)
)

removed_seasons = st.multiselect(
    "Select seasons to remove. Note: The season is the year the season ended (e.g., 2012 is the season 2011-2012).",
    options=[None, 2012, 2013, 2020, 2021],
    default=None
)
#selected_seasons = st.sidebar.slider("Select Seasons", min_value=min_season, max_value=max_season, value=(min_season, max_season))



#if 'All' in selected_seasons:
#    filtered_seasons = df['season'].unique()
#else:
#    filtered_seasons = selected_seasons

filtered_df = df[
    (df['season'].between(selected_seasons[0], selected_seasons[1])) & (~df['season'].isin(removed_seasons))
]

st.write("### Summary of Data")
st.write(f"There are a total of {filtered_df['season'].nunique()} included seasons: {filtered_df['season'].unique()}")

st.write("### Feature Engineering and Selection")
st.write(
    """
    While there could be countless features we might engineer and select from, we are particularly interested in
    the factors that Dean Oliver determined in his work *Basketball On Paper*. As a result, we will have only a few
    options: 
    1. Only a team's four offensive factors
    2. Only a team's four defensive factors
    3. A team's four offensive and four defensive factors
    4. A team's difference between the four offensive and four defensive factors
    """
)

selected_features = st.selectbox(
    "Choose a feature set",
    (
        "Offensive Four Factors Only (4 Features)",
        "Defensive Four Factors Only (4 Features)",
        "Offensive and Defensive Four Factors (8 Features)",
        "Differenced Four Factors (4 Features)"
    ),
    index=2,
    placeholder="Select a feature set..."
)
off_factors = ['EFG_PCT','FTA_RATE','TM_TOV_PCT','OREB_PCT']
def_factors = ['OPP_EFG_PCT','OPP_FTA_RATE','OPP_TOV_PCT','OPP_OREB_PCT']
diff_factors = [factor + '_d' for factor in off_factors]
if selected_features == "Offensive Four Factors Only (4 Features)":
    selected_features = off_factors
elif selected_features == "Defensive Four Factors Only (4 Features)":
    selected_features = def_factors
elif selected_features == "Offensive and Defensive Four Factors (8 Features)":
    selected_features = off_factors + def_factors
else:
    selected_features = diff_factors

st.write(f"We have selected the following {len(selected_features)} features: ")
st.write(f"{selected_features}")

model_data = df[[selected_target] + selected_features]

# Linear Regression Output
X = sm.add_constant(model_data.drop(selected_target, axis=1))

# Fit Linear Model
model_ff = sm.OLS(endog=model_data[selected_target], exog=X).fit()

# In-Sample Fitted Values
df['y_pred'] = model_ff.predict()
df['res'] = df[selected_target] - df['y_pred']

# Filter
filtered_preds = df.loc[filtered_df.index, 'y_pred']
filtered_res = df.loc[filtered_df.index, 'res']

st.write("## Model Output")
with st.expander(f"Estimated Model"):
    st.write(model_ff.summary())

with st.expander("Estimated Equation"):
    factor_param_dict = dict(zip(selected_features, model_ff.params[1:]))
    selected_target_esc = selected_target.replace("_","\\_")
    eqn = f"\\hat{{{selected_target_esc}}} = {model_ff.params[0]:.2f}"
    for key, value in factor_param_dict.items():
        key_esc = key.replace("_", "\\_")
        eqn += f" + {value:.2f} \\cdot {key_esc}"

    st.latex(eqn)

with st.expander("Interpretations"):
    if selected_features == off_factors:
        st.write("- Intercept does not have a useful interpretation.")
    if selected_features == def_factors:
        st.write("- Intercept does not have a useful interpretation.")
    if selected_features == off_factors + def_factors:
        st.write("- Intercept does not have a useful interpretation.")
    else:
        st.write(f"- For the average team, the expected {selected_target_name.lower()} is {model_ff.params[0]:.2f}.")
    st.write("While holding all other predictors constant...")
    for key, value in factor_param_dict.items():
        if value < 0:
            direction = "decrease"
            value = -value
        else:
            direction = "increase"
        st.write(f"- For an increase of 0.01 in {key}, we expect the team's {selected_target_name} to **{direction}** by {value:.2f}%.")

with st.expander("Explained Variation"):
    st.write(f"**R-Squared**: {model_ff.rsquared*100:.1f}% of the variation in {selected_target_name} is explained by *this* model.")
    st.write(f"**Adjusted R-Squared**: {model_ff.rsquared_adj*100:.1f}% of the variation in {selected_target_name} is explained by *this* model, accounting for the number of estimated parameters (i.e., model complexity).")

with st.expander("Statistical Significance"):
    st.write("While there are many issues with p-values, nonetheless, the general interpretation is as follows...")
    st.write("- Assuming there is no linear relationship between the predictor and target (in a model with other predictors), the probability of observing what we've observed in our data (or more extreme) is the p-value.")
    st.write("- If this p-value is very small (i.e., less than a prespecified signficance level), then it's highly unlikely that our initial assumption of no linear relationship was correct in the first place. In this case, we would 'reject the null hypothesis'.")
    st.write("For the coefficients in our model...")
    non_sig_idx = np.where((model_ff.pvalues[1:] >= .05)==True)[0]
    if len(non_sig_idx) > 0:
        st.write(f"- The predictors with coefficients that are not significantly different from zero: {selected_features[non_sig_idx]}")
    else:
        st.write(f"- All coefficients are signficantly different from zero (i.e., a significant linear relationship between the predictor and target exists given the other predictors in the model).")
# Visualizations
teams = ['All'] + list(df['TEAM_NAME'].unique())
selected_teams = st.sidebar.multiselect("Select Teams", options=teams, default='All')
if 'All' in selected_teams:
    filtered_teams = df['TEAM_NAME'].unique()
else:
    filtered_teams = selected_teams
    filtered_df = df[
        (df['TEAM_NAME'].isin(filtered_teams)) & (df['season'].between(selected_seasons[0], selected_seasons[1]))]
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

