#%% Imports
from math import ceil, floor
import pickle
import numpy as np
import streamlit as st
import pandas as pd

#%% MODEL LOADING
df = pd.read_csv(r"house_price.csv")

dropColumns = ["Id", "MSSubClass", "MSZoning", "Street", "LandContour", "Utilities", "LandSlope", "Condition1", "Condition2", "BldgType", "OverallCond", "RoofStyle",
               "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterCond", "Foundation", "BsmtCond", "BsmtExposure", "BsmtFinType1",
               "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF", "Heating", "Electrical", "LowQualFinSF", "BsmtFullBath", "BsmtHalfBath", "HalfBath"] + ["SaleCondition", "SaleType", "YrSold", "MoSold", "MiscVal", "MiscFeature", "Fence", "PoolQC", "PoolArea", "ScreenPorch", "3SsnPorch", "EnclosedPorch", "OpenPorchSF", "WoodDeckSF", "PavedDrive", "GarageCond", "GarageQual", "GarageType", "FireplaceQu", "Functional", "KitchenAbvGr", "BedroomAbvGr"]

droppedDf = df.drop(columns=dropColumns, axis=1)

droppedDf.isnull().sum().sort_values(ascending=False)
droppedDf["Alley"].fillna("NO", inplace=True)
droppedDf["LotFrontage"].fillna(df.LotFrontage.mean(), inplace=True)
droppedDf["GarageFinish"].fillna("NO", inplace=True)
droppedDf["GarageYrBlt"].fillna(df.GarageYrBlt.mean(), inplace=True)
droppedDf["BsmtQual"].fillna("NO", inplace=True)
droppedDf["MasVnrArea"].fillna(0, inplace=True)
droppedDf['MasVnrAreaCatg'] = np.where(droppedDf.MasVnrArea > 1000, 'BIG',
                                       np.where(droppedDf.MasVnrArea > 500, 'MEDIUM',
                                                np.where(droppedDf.MasVnrArea > 0, 'SMALL', 'NO')))

droppedDf = droppedDf.drop(['SalePrice'], axis=1)
inputDf = droppedDf.iloc[[0]].copy()

for i in inputDf:
    if inputDf[i].dtype == "object":
        inputDf[i] = droppedDf[i].mode()[0]
    elif inputDf[i].dtype == "int64" or inputDf[i].dtype == "float64":
        inputDf[i] = droppedDf[i].mean()

obj_feat = list(inputDf.loc[:, inputDf.dtypes == 'object'].columns.values)
for feature in obj_feat:
    inputDf[feature] = inputDf[feature].astype('category')

# load the model weights and predict the target
modelName = r"finalized_model.model"
loaded_model = pickle.load(open(modelName, 'rb'))

#%% STREAMLIT FRONTEND DEVELOPMENT

st.title("House Prices Prediction")
st.write("### This is a simple model for house prices prediction.")

st.sidebar.title("Model Parameters")
st.sidebar.write("### Categorical Features")

inputDict = dict(inputDf)

variables = droppedDf["Alley"].drop_duplicates().to_list()
inputDict["Alley"] = st.sidebar.selectbox("Alley", options=variables)
variables = droppedDf["Neighborhood"].drop_duplicates().to_list()
inputDict["Neighborhood"] = st.sidebar.selectbox(
    "Neighborhood", options=variables)
variables = droppedDf["HouseStyle"].drop_duplicates().to_list()
inputDict["HouseStyle"] = st.sidebar.selectbox("HouseStyle", options=variables)
variables = droppedDf["KitchenQual"].drop_duplicates().to_list()
inputDict["KitchenQual"] = st.sidebar.selectbox(
    "KitchenQual", options=variables)
variables = droppedDf["GarageFinish"].drop_duplicates().to_list()
inputDict["GarageFinish"] = st.sidebar.selectbox(
    "GarageFinish", options=variables)


st.sidebar.write("### Numeriacal Features")

inputDict["LotFrontage"] = st.sidebar.slider("LotFrontage", ceil(droppedDf["LotFrontage"].min()),
                                             floor(droppedDf["LotFrontage"].max()), int(droppedDf["LotFrontage"].mean()))

inputDict["OverallQual"] = st.sidebar.slider("OverallQual", ceil(droppedDf["OverallQual"].min()),
                                             floor(droppedDf["OverallQual"].max()), int(droppedDf["OverallQual"].mean()))

inputDict["YearBuilt"] = st.sidebar.slider("YearBuilt", ceil(droppedDf["YearBuilt"].min()),
                                           floor(droppedDf["YearBuilt"].max()), int(droppedDf["YearBuilt"].mean()))

inputDict["GrLivArea"] = st.sidebar.slider("GrLivArea", ceil(droppedDf["GrLivArea"].min()),
                                           floor(droppedDf["GrLivArea"].max()), int(droppedDf["GrLivArea"].mean()))

inputDict["TotalBsmtSF"] = st.sidebar.slider("TotalBsmtSF", ceil(droppedDf["TotalBsmtSF"].min()),
                                             floor(droppedDf["TotalBsmtSF"].max()), int(droppedDf["TotalBsmtSF"].mean()))

for key, value in inputDict.items():
    inputDf[key] = value

obj_feat = list(inputDf.loc[:, inputDf.dtypes == 'object'].columns.values)
for feature in obj_feat:
    inputDf[feature] = inputDf[feature].astype('category')

prediction = loaded_model.predict(inputDf)

st.write("### Prediction: ", prediction.item())

st.write("###### Group 2 | Week 4 - Machine Learning Model Deployment")
st.write("Date: 2020-05-20")
st.write("Version: 1.0")
