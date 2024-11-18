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
X_train,X_valid,y_train,y_valid = train_test_split(X_df,y,random_state = 12345,test_size = 0.7)
clf = LogisticRegression()
clf.fit(X_train,y_train)

intercept = pd.DataFrame(
    index = ["intercept"],
    columns = ["value"],
    data = clf.intercept_,
)

coef = pd.DataFrame(
    index = X_train.columns,
    columns = ["value"],
    data = clf.coef_.transpose(),
)
logit_coef_df = pd.concat([intercept,coef],axis = 0)
logit_coef_df = logit_coef_df.reset_index().sort_values("value")

st.title('ロジットモデルの係数')
fig = px.bar(data_frame = logit_coef_df.reset_index(),y = "index",x = "value" )
st.plotly_chart(fig)

st.write(logit_coef_df)

from sklearn.metrics import roc_auc_score

prob_train = clf.predict_proba(X_train)[:,1]
prob_valid = clf.predict_proba(X_valid)[:,1]

auc_train = roc_auc_score(y_train,prob_train)
auc_valid = roc_auc_score(y_valid,prob_valid)

st.write(auc_train)
st.write(auc_valid)
