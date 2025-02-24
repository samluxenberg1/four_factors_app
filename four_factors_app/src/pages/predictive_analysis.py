import streamlit as st

st.write("# Predictive Analysis of Four Factors")
st.write(
    """
    On this page, we will perform a predictive analysis using our four factor model. As our focus here is **not** on 
    the exact understanding of the relationships between factors, we instead will try to find the model that is most 
    predictive **of the future**. We can adapt this problem based on the granularity of our data. Theoretically, we
    can obtain four factor information at the game level. This would allow us to provide predictions of wins or win % 
    later in a given season. Alternatively, if we stick with the currently data that we used for the retrospective
    analysis (season-level information), we can use the previous season's four factors to predict what next season's
    results will be. Admittedly, the former is probably more useful than the latter because we would be able to take 
    into account team dynamics over time. In the latter case, we won't take into account any personnel (coaching or player)
    changes, or any other information that could impact team winning. 
    
    One of the most critical aspects of doing data science in the real world is thinking ahead. 
    As the data scientist, we want to be able to answer the following: 
     - What is the ultimate goal of the project?
     - Who will be the end-user of the work?
     - How will the end-user use the work?
     - What will this work help the end-user to achieve?
     
     ## Scenario
     Imagine an NBA season just ended and your favorite sports book has already put out odds of various end-of-season
     results for next season. 
     - Which team will have the best record overall?
     - Which teams will be the best in both the eastern and western conferences?
     - Which teams will win their division?
     - Which teams will make the playoffs?
     - Which team will have the highest odds of getting next year's number 1 overall pick in the draft? 
     
    ## Goal
    - Build the most predictive model we can. 
    
    ## Methodology
    - We will start with the four different methods of using the four factors as features. The critical piece to pay 
    attention to is the potential of data leakage. 
    """
)