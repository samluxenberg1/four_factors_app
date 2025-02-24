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

st.write("## Decisions")
with st.expander("Decisions"):
    st.write("""
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
    - How should we deal with teams that moved and/or changed names?
        - For simplicity, let's use the following team name conventions:
            - New Jersey Nets --> Brooklyn Nets
            - Charlotte Bobcats --> Charlotte Hornets
            - New Orleans/Oklahoma City Hornets and New Orleans Hornets --> New Oreleans Pelicans
            - Seattle SuperSonics --> Oklahoma City Thunder
            - LA Clippers --> Los Angeles Clippers
        
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
df.loc[df['TEAM_NAME'] == 'New Jersey Nets', 'TEAM_NAME'] = 'Brooklyn Nets'
df.loc[df['TEAM_NAME'] == 'Charlotte Bobcats', 'TEAM_NAME'] = 'Charlotte Hornets'
df.loc[df['TEAM_NAME'].isin(
    ['New Orleans/Oklahoma City Hornets', 'New Orleans Hornets']), 'TEAM_NAME'] = 'New Orleans Pelicans'
df.loc[df['TEAM_NAME'] == 'Seattle SuperSonics', 'TEAM_NAME'] = 'Oklahoma City Thunder'
df.loc[df['TEAM_NAME'] == 'LA Clippers', 'TEAM_NAME'] = 'Los Angeles Clippers'
# Process Data
X1_diff, y, y2 = data_processing.processing(df=df)
df = pd.concat([df, X1_diff.drop('const', axis=1)], axis=1)
st.dataframe(df.head().drop('TEAM_ID', axis=1))
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
off_factors = ['EFG_PCT', 'FTA_RATE', 'TM_TOV_PCT', 'OREB_PCT']
def_factors = ['OPP_EFG_PCT', 'OPP_FTA_RATE', 'OPP_TOV_PCT', 'OPP_OREB_PCT']
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
    selected_target_esc = selected_target.replace("_", "\\_")
    eqn = f"\\hat{{{selected_target_esc}}} = {model_ff.params[0]:.2f}"
    count = 0
    for key, value in factor_param_dict.items():
        key_esc = key.replace("_", "\\_")
        eqn += f" + {value:.2f} \\cdot {key_esc}"
        count += 1
        if count % 4 == 0:
            eqn += " \\\\ "

    st.latex(r"\small " + eqn)

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
        st.write(
            f"- For an increase of 0.01 in {key}, we expect the team's {selected_target_name} to **{direction}** by {value:.2f}%.")

with st.expander("Explained Variation"):
    st.write(
        f"**R-Squared**: {model_ff.rsquared * 100:.1f}% of the variation in {selected_target_name} is explained by *this* model.")
    st.write(
        f"**Adjusted R-Squared**: {model_ff.rsquared_adj * 100:.1f}% of the variation in {selected_target_name} is explained by *this* model, accounting for the number of estimated parameters (i.e., model complexity).")

with st.expander("Statistical Significance"):
    st.write("While there are many issues with p-values, nonetheless, the general interpretation is as follows...")
    st.write(
        "- Assuming there is no linear relationship between the predictor and target (in a model with other predictors), the probability of observing what we've observed in our data (or more extreme) is the p-value.")
    st.write(
        "- If this p-value is very small (i.e., less than a prespecified signficance level or acceptable frequency of type 1 errors), then it's highly unlikely that our initial assumption of no linear relationship was correct in the first place. In this case, we would *reject the null hypothesis*.")
    st.write("For the coefficients in our model...")
    non_sig_idx = np.where((model_ff.pvalues[1:] >= .05) == True)[0]
    if len(non_sig_idx) > 0:
        st.write(
            f"- The predictors with coefficients that are not significantly different from zero: {selected_features[non_sig_idx]}")
    else:
        st.write(
            f"- All coefficients are signficantly different from zero (i.e., a significant linear relationship between the predictor and target exists given the other predictors in the model).")
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
st.write("### Assumption Analysis")
st.write(
    f"""
    While we can certainly estimate a model, interpret its coefficients, and perform statistical inference, we can't 
    really trust any of this unless we validate our model assumptions. For the linear regression model, there are 
    quite a few assumptions to check. 
    
    1. Random errors are assumed to be independent and come from a normal distribution of zero mean and constant variance.  
    2. We assume that we can represent the relationship between the {selected_target_name} and its predictors with a linear model (i.e., linearity).
    3. We assume there are no highly influential points or outliers. 

    """
)
with st.expander("Residuals vs. Predicted"):
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    sns.scatterplot(x=filtered_preds, y=filtered_res, alpha=.5, ax=ax2)
    ax2.axhline(y=0, color='red', linestyle='dashed')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')

    st.pyplot(fig2)

    st.write(
        "In this plot, we are looking for a random scattering of points with no obvious patterns. Such a plot can help us to diagnose non-linearity, dependence, and non-constant variance. ")

with st.expander("Residuals vs. Features"):
    n = int(np.ceil(np.sqrt(len(selected_features))))
    fig3, ax3 = plt.subplots(n, n, figsize=(14, 14))
    ax3 = ax3.ravel()
    for idx, feature in enumerate(selected_features):
        ax = ax3[idx]
        sns.scatterplot(x=filtered_df[feature], y=filtered_res, alpha=.5, ax=ax)
        ax.axhline(y=0, color='red', linestyle='dashed')
        ax.set_xlabel(f'{feature}')
        ax.set_ylabel('Residuals')
        ax.set_title(f'Residuals vs. {feature}')

    # Remove unused subplots
    for j in range(idx + 1, len(ax3)):
        fig3.delaxes(ax3[j])

    # Adjust layout
    fig3.tight_layout()

    st.pyplot(fig3)

    st.write(
        "Similar to the previous residual plot, here, we are looking for a random scattering of points with no obvious patterns. Such a plot can help us to diagnose non-linearity, dependence, and non-constant variance. ")

with st.expander("Q-Q Plot"):
    st.write(
        "Note that while we could test for normality using a variety of hypothesis tests, it's generally recommended to simply examine the Q-Q plots.")
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    sm.qqplot(filtered_res, line='q', ax=ax4, alpha=.5)
    st.pyplot(fig4)
    st.write(
        "In this plot, we're looking for obvious deviations from the main diagonal. This would point to non-normalities; however, minor departures may not be a big deal.")

st.write("### Final Metrics and Interpretations")
with st.expander("Model Metrics"):
    mse = model_ff.mse_resid
    rmse = mse ** .5
    mae = np.mean(np.abs(model_ff.resid))
    if selected_target == "W":
        units = "wins"
        st.write(
            f"""
                - Mean Squared Error (MSE): {float(mse): .3f}
                - Root Mean Squared Error (RMSE): {float(rmse): .3f}
                - Mean Average Error (MAE): {float(mae): .3f}

                On average, our model's (in-sample) estimates of {selected_target_name} are off by {float(min(mae, rmse)): .2f} to {float(max(mae, rmse)): .2f} {selected_target_name.lower()}.
                """
        )
    else:
        units = "% (percentage points)"
        rmse = rmse * 100
        mae = mae * 100
        st.write(
            f"""
            - Root Mean Squared Error (RMSE): {float(rmse): .2f}%
            - Mean Average Error (MAE): {float(mae): .2f}%
            
            On average, our model's (in-sample) estimates of {selected_target_name} are off by {float(min(mae, rmse)): .2f}% to {float(max(mae, rmse)): .2f}%. 
            We can translate this into wins by multiplying by 82 games. On average, our estimates are off by {float(min(mae, rmse) * 82 / 100): .2f} wins to {float(max(mae, rmse) * 82 / 100): .2f} wins.
            """
        )

st.write("## Actionable Insights")
st.write(
    """
    All of the above statistical analysis is great. However, we *cannot* simply dump these results onto a coaching staff. 
    Most members of our coaching staff do not have a technical background. They are the basketball (i.e., domain) experts.
    As the data scientists on the team, we need to translate these results into something they can act on and use. 
    
    Let's think back to our scenario and the original goal of all of this analysis. Our goal is to make recommendations
    to the coaching staff on what to improve during the off-season. We can clearly see that efficient shooting and 
    taking care of the ball are the most important factors that lead to winning. We can certainly suggest that overall
    these are the two aspects that teams should work on. However, each team is different. Some teams may already be
    elite in these areas and weak in the free-throw rate and offensive rebound percentage. Therefore, it might be better
    to make a plan to improve the weaknesses of the team while maintaining its current strengths. 
    """
)
st.write("### Task")
st.write("""
Explore a particular team's most recent numbers and make recommendations for that team's coaching staff based on 
the factors at which they are the **weakest**.
""")

selected_team_insights = st.selectbox("Select Team", options=df['TEAM_NAME'].unique())
last_season = df['season'].max()
df_team_insights = df[(df['TEAM_NAME'] == selected_team_insights) & (df['season'] == last_season)][
    selected_features + [selected_target, 'y_pred']]
df_team_insights_display = df_team_insights.drop(selected_target, axis=1).T.reset_index()
df_team_insights_display.columns = ['Feature', f'{selected_team_insights}']

# Compute League Values for Comparison
df_league = df[df['season'] == last_season][selected_features + ['y_pred']]
df_league_summary = pd.DataFrame()

df_league_summary['League Avg'] = df_league.mean()
df_league_summary['League 10th%'] = np.percentile(df_league, 10, axis=0)
df_league_summary['League 25th%'] = np.percentile(df_league, 25, axis=0)
df_league_summary['League 50th%'] = np.percentile(df_league, 50, axis=0)
df_league_summary['League 75th%'] = np.percentile(df_league, 75, axis=0)
df_league_summary['League 90th%'] = np.percentile(df_league, 90, axis=0)
df_league_summary = df_league_summary.reset_index().rename(columns={'index': 'Feature'})


# Compute Selected Team's Percentiles
def extract_percentile(df_league, df_team, feature):
    team_value = df_team.loc[df_team['Feature'] == feature, f'{selected_team_insights}'].values
    feature_vals = df_league[feature].values
    return np.round((feature_vals <= team_value).mean() * 100, 2)


df_team_insights_display['Team %'] = df_team_insights_display.apply(lambda x: extract_percentile(
    df_league=df_league,
    df_team=df_team_insights_display,
    feature=x['Feature']
),
                                                                    axis=1
                                                                    )
# Make Prediction
df_y_team = pd.DataFrame({
    f'Actual {selected_target_name}': df_team_insights[selected_target],
    f'Estimated {selected_target_name}': df_team_insights['y_pred']
}).reset_index(drop=True)

st.dataframe(df_y_team)
df_team_insights_display_join = df_team_insights_display.merge(
    df_league_summary,
    how='left',
    on='Feature'
)
df_team_insights_display_join.loc[
    df_team_insights_display_join['Feature'] == 'y_pred', 'Feature'] = f'Est. {selected_target_name}'
st.dataframe(df_team_insights_display_join)

lower_is_worse_off_factors = ['EFG_PCT', 'FTA_RATE', 'OREB_PCT']
higher_is_worse_off_factors = ['TM_TOV_PCT']
lower_is_worse_def_factors = ['OPP_TOV_PCT']
higher_is_worse_def_factors = ['OPP_EFG_PCT', 'OPP_FTA_RATE', 'OPP_OREB_PCT']
lower_is_worse_diff_factors = ["EFG_PCT_d", "FTA_RATE_d", "OREB_PCT_d"]
higher_is_worse_diff_factors = ["TM_TOV_PCT_d"]

st.write("""
### Selecting percentiles for making recommendations
#### Lower Percentiles
 - Certain features have a positive impact when they increase numerically. At the same time, they also have a negative
 impact when they decrease. For example, lower values of your team's eFG % suggests that your team is inefficient in its
 scoring. 
 - The lower percentile threshold represents the value below which you would be comfortable suggesting an area to focus.
 - The default value is 30. This means that when your team is below the 30th percentile with respect to 
 the entire league for a given feature, this feature is in contention for an area to focus on in the off-season. 
 - The suggested area of focus is the one with the lowest percentile. 
 
 #### Higher Percentiles
 - Certain features have a positive impact when they decrease numerically. At the same time, they also have a negative
 impact when they increase. For example, higher values of your team's tunover % suggests that your team does not take
 care of the ball. 
 - THe higher percentile threshold represents the value above which you would be comfortable suggesting an area to focus.
 - The default value is 70. This means that when your team is above the 70th percentile with respect to the entire league 
 for a given feature, this feature is in contention for an area to focus on in the off-season. 
 - The suggested area of focus is the one with the highest percentile. 
 
 **NOTE: This is the the only way to choose among potential recommendations. In fact, this probably isn't even the best way
 to go about such a decision. More likely, the team would probably want to focus on areas that will make the largest impact, 
 even if they're not among the extreme percentiles. For example, improving the team's effective field goal percentage 
 will likely be more impactful for most teams than improving on defensive rebounding (to counteract the opponents' 
 offensive rebounding percentage.)**  
""")
col1, col2 = st.columns(2)
with col1:
    lower_thresh = st.number_input("Select a lower percentile", value=30, max_value=100, min_value=0)
with col2:
    higher_thresh = st.number_input("Select a higher percentile", value=70, max_value=100, min_value=0)
def make_recommendation(df, lower, higher, lower_thresh, higher_thresh):
    df = df.copy()
    df_lower = df[df['Feature'].isin(lower)].reset_index()
    df_higher = df[df['Feature'].isin(higher)].reset_index()
    lower_idx = df_lower['Team %'].idxmin()
    higher_idx = df_higher['Team %'].idxmax()

    if df_lower['Team %'].iloc[lower_idx] < lower_thresh:
        st.write(f"""
        - Increasing {df_lower['Feature'].iloc[lower_idx]}.
        """)
    if df_higher['Team %'].iloc[higher_idx] > higher_thresh:
        st.write(f"""
        - Decreasing {df_higher['Feature'].iloc[higher_idx]}.
        """)
    if (df_lower['Team %'].iloc[lower_idx] >= lower_thresh) & (df_higher['Team %'].iloc[higher_idx] <= higher_thresh):
        st.write(f"""
        - Nothing in particular.
        """)
    st.write(f"""
        Are there any other candidates for areas of focus for this upcoming off-season?
        """)


with st.expander("Recommendation"):
    df_rec = df_team_insights_display_join.copy()
    st.write(f"{selected_team_insights} should focus on ...")
    if selected_features == off_factors:
        make_recommendation(
            df=df_rec,
            lower=lower_is_worse_off_factors,
            higher=higher_is_worse_off_factors,
            lower_thresh=lower_thresh,
            higher_thresh=higher_thresh
        )
    if selected_features == def_factors:
        make_recommendation(
            df=df_rec,
            lower=lower_is_worse_def_factors,
            higher=higher_is_worse_def_factors,
            lower_thresh=lower_thresh,
            higher_thresh=higher_thresh
        )
    if selected_features == off_factors + def_factors:
        make_recommendation(
            df=df_rec,
            lower=lower_is_worse_off_factors + lower_is_worse_def_factors,
            higher=higher_is_worse_off_factors + higher_is_worse_def_factors,
            lower_thresh=lower_thresh,
            higher_thresh=higher_thresh
        )
    if selected_features == diff_factors:
        make_recommendation(
            df=df_rec,
            lower=lower_is_worse_diff_factors,
            higher=higher_is_worse_diff_factors,
            lower_thresh=lower_thresh,
            higher_thresh=higher_thresh
        )
