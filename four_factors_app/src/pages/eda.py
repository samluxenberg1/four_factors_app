import os
import sys
from pathlib import Path

# Add root directory to sys.path before importing your modules
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))
print(f"Root Directory: {root_dir}")

from src import data_processing
from src.utils import widget_filters
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from io import BytesIO
import streamlit as st

st.write(
    """
    # Exploratory Data Analysis 
    Exploratory data analysis (EDA) is useful in data science projects for multiple reasons. 
    - It allows the data scientist to get an understanding of the general distributions in the data. 
    - It can inform which parts of the data need more consideration (e.g., outliers, missing values).
    - It can answer questions on how to approach modeling.
    
    For a predictive modeling project, EDA should technically be done only on training data, as we don't want any
    biases sneaking into our testing data and analysis. 
    
    For more of a classical statistical inference perspective, where we are particularly interested in inference and 
    understanding the relationships between our variables, there is not always the need to split data between 
    training and testing. Nonetheless, EDA is still a critical component of such a project.   
    """
)

df = st.session_state["data"]

# Process Data
X1_diff, y, y2 = data_processing.processing(df=df)
df = pd.concat([df, X1_diff.drop('const', axis=1)], axis=1)

teams = ['All'] + list(df['TEAM_NAME'].unique())
seasons = ['All'] + list(df['season'].unique())
min_season = int(df['season'].min())
max_season = int(df['season'].max())

# Sidebar Widgets
selected_target = st.sidebar.selectbox(
    "Choose your target (i.e. response) variable.",
    ("Wins", "Win %"),
    index=1,
    placeholder="Select target variable..."
)
if selected_target == "Wins":
    target = "W"
else:
    target = "W_PCT"


selected_features = st.sidebar.selectbox(
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


st.write(f"## Team Factors Over Time")
st.write(
    """
    These graphs depict the trends in team offensive factors from season to season relative to the league average. By 
    filtering for specific teams, we can see how that team has performed historically relative to league average. That 
    could potentially provide insights into how team performance waxed or waned historically. With this information in 
    hand, we might begin to ask deeper and more important questions about why team factors changed over time:
     - Was it due to the acquisition of a specific player? 
     - Was it due to league-wide trends? 
     - Was it due to changes in the coaching staff?
     - Was it due to changes in league rules?      
    """
)

# filtered_df_fig1, selected_teams_fig1, selected_seasons_fig1 = config_widgets(
#     df=df,
#     min_season=min_season,
#     max_season=max_season,
#     teams=teams,
#     key="fig1"
# )
col1, col2 = st.columns(2)
with col1:
    selected_teams_fig1 = st.multiselect(
            "Select Teams",
            options=teams,
            default='All',
            key="teams_1"
    )
with col2:
    selected_seasons_fig1 = st.slider(
            "Select Seasons",
            min_value=min_season,
            max_value=max_season,
            value=(min_season, max_season),
            key="seasons_1"
    )
filtered_df_fig1 = widget_filters(
    selected_teams=selected_teams_fig1,
    selected_seasons=selected_seasons_fig1,
    df=df
)
feature_columns = X1_diff.drop('const', axis=1).columns.tolist()

fig0, ax0 = plt.subplots(2, 2, figsize=(13, 12), dpi=90)
ax0 = ax0.ravel()

for idx, feature in enumerate(off_factors):
    ax = ax0[idx]

    # Compute League Average by Season Using Filtered data
    league_avg = df.groupby('season')[feature].agg('mean').reset_index()
    filtered_league_avg = league_avg[league_avg['season'].between(selected_seasons_fig1[0], selected_seasons_fig1[1])]
    # Plot scatter plot for each selected team in filtered data
    for team in filtered_df_fig1['TEAM_NAME'].unique():
        team_df = filtered_df_fig1[filtered_df_fig1['TEAM_NAME'] == team]
        if filtered_df_fig1['TEAM_NAME'].nunique() > 4:
            sns.scatterplot(x=team_df['season'], y=team_df[feature], ax=ax, color='gray', alpha=.5)
        else:
            sns.scatterplot(x=team_df['season'], y=team_df[feature], ax=ax, label=f"{team}", alpha=.5)

    # Plot line for league average
    sns.lineplot(x=filtered_league_avg['season'].unique(), y=filtered_league_avg[feature], ax=ax, label='League Avg',
                 color='red')


    ax.set_title(f"{feature} Over Seasons")
    ax.set_xlabel("")
    ax.set_xticks(filtered_league_avg['season'])
    ax.set_xticklabels(filtered_league_avg['season'].astype(int), rotation=45)

    ax.set_ylabel(feature)
    ax.legend(loc='best')


buf = BytesIO()
fig0.savefig(buf, format="png")
st.image(buf)
#plt.tight_layout()
#st.pyplot(fig0)

# Pair Plot with Correlation
# Fig 2 - Pair plots w/ Correlation
st.write(f"## Bivariate and Univariate Distributions Among Team Factors and {selected_target}")
st.write(
    f"""
    There are multiple aspects to this figure. Along the diagonal, we see the distributions of each factor and the 
    {selected_target}. Below the diagonal, we have the relationship between each factor and the {selected_target}. 
    These plots include a fitted linear regression line that may indicate the directionality and magnitude of the 
    bivariate relationships. Above the diagonal, we have all pearson (linear) correlations for pair of variables. 
    """
)
col1, col2 = st.columns(2)
with col1:
    selected_teams_fig2 = st.multiselect(
            "Select Teams",
            options=teams,
            default='All',
            key="teams_2"
    )
with col2:
    selected_seasons_fig2= st.slider(
            "Select Seasons",
            min_value=min_season,
            max_value=max_season,
            value=(min_season, max_season),
            key="seasons_2"
    )
filtered_df_fig2 = widget_filters(
    selected_teams=selected_teams_fig2,
    selected_seasons=selected_seasons_fig2,
    df=df
)
# filtered_df_fig2, selected_teams_fig2, selected_seasons_fig2 = config_widgets(
#     df=df,
#     min_season=2002,
#     max_season=2024,
#     teams=teams,
#     key="fig2"
# )
def reg_coef(x, y, label=None, color=None, **kwargs):
    ax = plt.gca()
    r, p = pearsonr(x, y)
    ax.annotate(f'r = {r:.2f}', xy=(.5, .5), xycoords='axes fraction', ha='center')
    ax.set_axis_off()


df_ff = filtered_df_fig2[selected_features + [target]]
g = sns.PairGrid(df_ff, height=1.75, aspect=1) #aspect = 1.35, height=2.25
g.map_diag(sns.histplot, kde=True)
g.map_lower(sns.regplot, scatter_kws={'alpha': .25})
g.map_upper(reg_coef)

buf = BytesIO()
g.figure.savefig(buf, format="png")
st.image(buf)
#st.pyplot(g.figure)

# Correlation Over Time
# Fig 3 - Pair plots w/ Correlation
st.write(f"## {selected_target} Correlation Over Time")
st.write(
    f"""
    The plot below shows how the correlation between the four team factors and {selected_target} changes from season to season. 
    Note that this plot will only display when **more than 1** team is selected. After all, we cannot compute correlation
    with a single data point. It is likely only useful to view this graphic with `All` teams selected or defined groups
    of teams (e.g., divisions, conferences, etc.).  
    """
)
col1, col2 = st.columns(2)
with col1:
    selected_teams_fig3 = st.multiselect(
            "Select Teams",
            options=teams,
            default='All',
            key="teams_3"
    )
with col2:
    selected_seasons_fig3= st.slider(
            "Select Seasons",
            min_value=min_season,
            max_value=max_season,
            value=(min_season, max_season),
            key="seasons_3"
    )
filtered_df_fig3 = widget_filters(
    selected_teams=selected_teams_fig3,
    selected_seasons=selected_seasons_fig3,
    df=df
)
# filtered_df_fig3, selected_teams_fig3, selected_seasons_fig3 = config_widgets(
#     df=df,
#     min_season=2002,
#     max_season=2024,
#     teams=teams,
#     key="fig3"
# )
corr_data = []
for s in filtered_df_fig3['season'].unique():
    # Filter data for specific season
    df_season = filtered_df_fig3[filtered_df_fig3['season']==s]

    # Compute correlations for each factor with target
    corr_vals = [pearsonr(df_season[factor], df_season[target])[0] for factor in selected_features]

    # Append results as a dictionary
    corr_data.append(
        {'season': s, **{factor: r for factor, r in zip(selected_features, corr_vals)}}
    )

# Convert the list of dictionaries into a dataframe
df_time_corr = pd.DataFrame(corr_data)
df_time_corr = df_time_corr[['season']+selected_features]

fig_corr, ax_corr = plt.subplots(figsize=(15,7), dpi=80)
for idx, feat in enumerate(selected_features):
    sns.lineplot(x='season', y=feat, data=df_time_corr, label=feat, ax=ax_corr, alpha=.75)
#sns.lineplot(x='season', y=off_factors[0], data=df_time_corr, label=off_factors[0], ax=ax_corr, alpha=.75)
#sns.lineplot(x='season', y=off_factors[1], data=df_time_corr, label=off_factors[1], ax=ax_corr, alpha=.75)
#sns.lineplot(x='season', y=off_factors[2], data=df_time_corr, label=off_factors[2], ax=ax_corr, alpha=.75)
#sns.lineplot(x='season', y=off_factors[3], data=df_time_corr, label=off_factors[3], ax=ax_corr, alpha=.75)
ax_corr.set_ylim(-1,1)
ax_corr.set_title('Team Factor Correlation Over Time')
ax_corr.set_xlabel('Season')
ax_corr.set_ylabel('Pearson Correlation')

buf = BytesIO()
fig_corr.savefig(buf, format="png")
st.image(buf)
#st.pyplot(fig_corr)


