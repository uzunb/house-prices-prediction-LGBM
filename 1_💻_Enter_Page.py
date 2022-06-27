import streamlit as st
from streamlit.logger import get_logger
import datetime

LOGGER = get_logger(__name__)

thedate = datetime.date.today()
def run():
    st.image('./clear-glass-large-windows-on-a-house-with-asymmetrical-cube-shaped-structures-built-near-an-oval-blue-pool.jpg')
    # st.set_page_config(page_title="Enter Page", page_icon="💻")

    st.write("""
    # Welcome to House Price Prediction!
    """) 

    st.markdown(
        """
    This repo has been developed for the Istanbul Data Science Bootcamp, organized in cooperation with IBB & Kodluyoruz. 
    Prediction for house prices was developed using the Kaggle House Prices - Advanced Regression Techniques competition dataset.
    
    ### Goal

    The goal of this project is to predict the price of a house in Ames using the features provided by the dataset.
    
    ------

    ###### Group 2 | Machine Learning Model Deployment

    ###### Version: 1.0
    """
    )
    st.write("###### Date: ", thedate)


if __name__ == "__main__":
    run()