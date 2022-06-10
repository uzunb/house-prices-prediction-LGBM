import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(page_title="Enter Page", page_icon="💻")

    st.write("""
    # Welcome to House Price Prediction!
    """) 

    st.markdown(
        """
    This repo has been developed for the Istanbul Data Science Bootcamp, organized in cooperation with İBB & Kodluyoruz. 
    Prediction for house prices was developed using the Kaggle House Prices - Advanced Regression Techniques competition dataset.
    
    ## Goal

    The goal of this project is to predict the price of a house in Ames using the features provided by the dataset.
    
    ------

    ###### Group 2 | Week 4 - Machine Learning Model Deployment
    ###### Date: 2020-05-20
    ###### Version: 1.0
    """
    )
    


if __name__ == "__main__":
    run()