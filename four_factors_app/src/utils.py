import pandas as pd
from pathlib import Path
import os
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(ROOT_DIR, 'data')


def load_data():
    data_path = os.path.join(DATA_DIR, "four_factors.csv")
    df = pd.read_csv(data_path)
    return df


def config_widgets(df: pd.DataFrame, min_season, max_season, teams, key):
    teams_key = f"{key}_teams"
    seasons_key = f"{key}_seasons"
    st.write(f"Keys: {teams_key} - {seasons_key}")
    col1, col2 = st.columns(2)
    with col1:
        selected_teams = st.multiselect(
            "Select Teams",
            options=teams,
            default='All',
            key=teams_key
        )
    with col2:
        selected_seasons = st.slider(
            "Select Seasons",
            min_value=min_season,
            max_value=max_season,
            value=(min_season, max_season),
            key=seasons_key
        )
    if "All" in selected_teams:
        filtered_teams = df['TEAM_NAME'].unique()
    else:
        filtered_teams = selected_teams

    filtered_df = df[(df['TEAM_NAME'].isin(filtered_teams)) & (df['season'].between(
        selected_seasons[0], selected_seasons[1]
    ))]

    return filtered_df, selected_teams, selected_seasons

def widget_filters(selected_teams, selected_seasons, df):
    if "All" in selected_teams:
        filtered_teams = df['TEAM_NAME'].unique()
    else:
        filtered_teams = selected_teams

    filtered_df = df[(df['TEAM_NAME'].isin(filtered_teams)) & df['season'].between(selected_seasons[0],selected_seasons[1])]

    return filtered_df


