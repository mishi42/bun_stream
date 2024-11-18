import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import plotly.express as px
import pygwalker as pyg
import streamlit.components.v1 as stc # streamlit 機能を拡張

X = load_breast_cancer()["data"]
y = load_breast_cancer()["target"]
feat = load_breast_cancer()["feature_names"]

X_df = pd.DataFrame(X,columns = feat)

pyg_html = pyg.walk(dataset = X_df,return_html = True)

stc.html(pyg_html, scrolling=True, height=1000) #CSS pixels

