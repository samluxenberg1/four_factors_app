import streamlit as st
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from four_factors_app.src.utils import load_data

if "data" not in st.session_state:
    st.session_state["data"] = load_data()

st.set_page_config(page_title="Four Factors Model", layout="wide")

st.title("Basketball Four Factor Exploration")
st.write("""
    The purpose of this app is to help students think about the full data science project workflow in industry. While 
    the details of the project correspond to the Four Factor Model in basketball (created by Dean Oliver and presented  
    in _Basketball On Paper_ and further discussed in his latest book _Basketball Beyond Paper_), this app will try to 
    mimic many of the most common aspects that need consideration across any industry. One major point of emphasis will  
    be to always think about the stakeholder and/or end-user at every stage of a data science project: project planning, 
    data collection, processing, feature engineering, modeling, analyzing, deployment, and monitoring.  
     
    Created by Dean Oliver and introduced in _Basketball On Paper_, the Four Factor Model in basketball describes what 
    goes into winning basketball games. The model draws its power from its simplicity and effectiveness. At its core, 
    it is simply a linear regression model that describes the strengths and weaknesses of teams. This app explores this 
    model from two perspectives. 

    - First, we will explore the four factors from an explanatory point of view (i.e., a retrospective analysis). This will allow us to explain past 
    performance. This might help coaches to determine what to work on in practice sessions to improve their own team 
    performance. At the same time, it might help them to prepare how to approach their next opponent based on their past 
    performance.     

    - Second, we will take a more predictive perspective. Instead of worrying about how to explain what happened in the 
    past based on these factors, we will focus on predicting what will happen in the future.  
    """)

st.write("This is the main dashboard. Select a page from the sidebar.")

# Sidebar Navigation
#page = st.sidebar.selectbox(
#    "Select a Page",
#    ["EDA","Retrospective Analysis", "Predictive Analysis"]
#)

#if page == "EDA":
#    import pages.eda
#elif page == "Retrospective Analysis":
#    import pages.retrospective_analysis
#elif page == "Predictive Analysis":
#    import pages.predictive_analysis