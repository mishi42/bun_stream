import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import plotly.express as px

X = load_breast_cancer()["data"]
y = load_breast_cancer()["target"]
feat = load_breast_cancer()["feature_names"]

X_df = pd.DataFrame(X,columns = feat)
X_df["target"] = y

st.write(X_df)

st.write(X_df.describe().T)
