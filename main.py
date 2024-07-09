import pandas as pd
from prophet import Prophet
import numpy as np
import matplotlib.pyplot as plt
import datetime
import jdatetime
from matplotlib.style import use
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import mean_absolute_percentage_error
from warnings import filterwarnings
import streamlit as st
import plotly.graph_objects as go




@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


def to_georgian(date):
    date  = date.split("/")
    date = jdatetime.date(year=int(date[0]), month=int(date[1]), day=int(date[2])).togregorian()
    return date


def to_jalali(row):
    d = jdatetime.date.fromgregorian(year=row.year, month=row.month, day=row.day)
    return f"{d.year}/{d.month}"


st.write("""
         # Prophet
         """)
st.sidebar.write("Controls")
file = st.sidebar.file_uploader("Uppload Your Dataset", type=".csv")

# df = pd.read_csv("SalesData.csv") if file is None else pd.read_csv(file)
try:
    df = pd.read_csv(file)
    got_data = True
except: got_data = False
if got_data:
    products = list(df.GoodName.unique())
    product = st.sidebar.selectbox(label="Please select a product", options=products)
    horizon = int(st.sidebar.slider(label="Select Prediction Horizon", min_value=2, max_value=30, value=5))
    test_size_manual = st.sidebar.number_input(label="Select Test Size", min_value=0, max_value=30, value=0)
    manual = st.sidebar.checkbox("Manual Mode")
    
    changepoint_prior_scale_manual=float(st.sidebar.text_input(label="Select changepoint_prior_scale", value=1))
    seasonality_prior_scale_manual=float(st.sidebar.text_input(label="Select seasonality_prior_scale", value=1))

    
    st.write(product)

    df_t = df.query(f"GoodName == '{product}'").reset_index(drop=True)

    df_t["Year"] = df_t["StrFactDate"].apply(lambda d: int(d.split("/")[0]))
    df_t["Month"] = df_t["StrFactDate"].apply(lambda d: int(d.split("/")[1]))
    dg = df_t.groupby(["Year", "Month"]).agg({"SaleAmount":"sum"}).reset_index()\
        .sort_values(["Year","Month"],ignore_index=True)
    dg["ds"] = (dg["Year"].astype(str)+ "/"+dg["Month"].astype(str) + "/01").apply(to_georgian)
        
    dg = dg[["ds", "SaleAmount"]].rename(columns={"SaleAmount":"y"})
    # dg
    if test_size_manual == 0:
        train_size = -10 if len(dg)>20 else -2
    else:
        train_size = -test_size_manual
    ds_train = dg.iloc[:train_size]
    ds_test = dg.iloc[train_size:]
    if not manual:
        def hyperparameter_tuning(space):
            model = Prophet(changepoint_prior_scale=space["changepoint_prior_scale"],
                            seasonality_prior_scale=space["seasonality_prior_scale"])
            model.fit(ds_train)
            future = pd.DataFrame(dg["ds"])
            future.index = future["ds"]
            future.drop("ds", axis=1)
            pred = model.predict(future)
            pred_data = pred.iloc[train_size:]
            mse = mean_absolute_percentage_error(ds_test["y"], pred_data["yhat"])
            return {'loss': mse, 'status': STATUS_OK, 'model': model}

        space = {
            'changepoint_prior_scale': hp.uniform("changepoint_prior_scale", 0.0001, 20),
            'seasonality_prior_scale': hp.uniform('seasonality_prior_scale', 0.001, 20)
        }

        trials = Trials()
        best = fmin(fn=hyperparameter_tuning,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=40,
                    trials=trials)
    else:
        best = {"changepoint_prior_scale":changepoint_prior_scale_manual,
                "seasonality_prior_scale":seasonality_prior_scale_manual}

    train_size = len(dg) + train_size

    model = Prophet(changepoint_prior_scale=best["changepoint_prior_scale"],
                    seasonality_prior_scale=best["seasonality_prior_scale"],
                    growth="linear")
    model.fit(ds_train)
    future = pd.DataFrame(dg["ds"])
    future.index = future["ds"]
    future.drop("ds", axis=1)
    pred = model.predict(future)

    ticks = dg["ds"].apply(to_jalali)
    st.dataframe(best)
    fig_tuned = go.Figure()
    fig_tuned.add_trace(go.Scatter(x=ticks, y=dg["y"], mode='markers', name='observations', marker=dict(color='black')))
    fig_tuned.add_trace(go.Scatter(x=ticks, y=pred["yhat"], mode='lines', name='predictions'))
    fig_tuned.add_trace(go.Scatter(x=ticks, y=pred["yhat_upper"], fill=None, mode='lines', line_color='lightgrey', showlegend=False))
    fig_tuned.add_trace(go.Scatter(x=ticks, y=pred["yhat_lower"], fill='tonexty', mode='lines', line_color='lightgrey', showlegend=False))
    fig_tuned.add_vline(x=ticks[train_size], line=dict(dash='dash', color='black'), name='split')

    fig_tuned.update_layout(title='Tuned Model Predictions', xaxis_title='Date', yaxis_title='Sales Amount')

    model = Prophet()
    model.fit(ds_train)
    future = pd.DataFrame(dg["ds"])
    future.index = future["ds"]
    future.drop("ds", axis=1)
    pred = model.predict(future)

    fig_vanilla = go.Figure()
    fig_vanilla.add_trace(go.Scatter(x=ticks, y=dg["y"], mode='markers', name='observations', marker=dict(color='black')))
    fig_vanilla.add_trace(go.Scatter(x=ticks, y=pred["yhat"], mode='lines', name='predictions'))
    fig_vanilla.add_trace(go.Scatter(x=ticks, y=pred["yhat_upper"], fill=None, mode='lines', line_color='lightgrey', showlegend=False))
    fig_vanilla.add_trace(go.Scatter(x=ticks, y=pred["yhat_lower"], fill='tonexty', mode='lines', line_color='lightgrey', showlegend=False))
    fig_vanilla.add_vline(x=ticks[train_size], line=dict(dash='dash', color='black'), name='split')

    fig_vanilla.update_layout(title='Vanilla Model Predictions', xaxis_title='Date', yaxis_title='Sales Amount')

    st.plotly_chart(fig_tuned)
    st.plotly_chart(fig_vanilla)

    # horizon = int(input("select prediction hoirizon"))
    model = Prophet(changepoint_prior_scale=best["changepoint_prior_scale"],
                seasonality_prior_scale=best["seasonality_prior_scale"],
                # growth = ["linear", "flat"][best["growth"]]
                growth = "linear"
                )


    model.fit(dg)
    future = [dg["ds"].max() + datetime.timedelta(days= 30*(i+1)) for i in range(horizon)]
    future = pd.DataFrame({"ds": future})
    pred = model.predict(future)
    fig_final = go.Figure()
    ticks = pred["ds"].apply(to_jalali)
    fig_final.add_trace(go.Scatter(x=ticks, y=pred["yhat"], mode='lines', name='predictions'))
    fig_final.add_trace(go.Scatter(x=ticks, y=pred["yhat_upper"], fill=None, mode='lines', line_color='lightgrey', showlegend=False))
    fig_final.add_trace(go.Scatter(x=ticks, y=pred["yhat_lower"], fill='tonexty', mode='lines', line_color='lightgrey', showlegend=False))
    fig_final.update_layout(title=f'Predictions for {horizon} months', xaxis_title='Date', yaxis_title='Sales Amount')
    st.plotly_chart(fig_final)
    df_final = pd.DataFrame({"Date": ticks, "Yhat":pred["yhat"], "Yhat_upper":pred["yhat_upper"], "Yhat_lower": pred["yhat_lower"]})


    csv = convert_df(df_final)

    st.download_button(
    "Download Results",
    csv,
    "file.csv",
    "text/csv",
    key='download-csv'
    )


else:
    st.write("Please uppload your data")
    df = pd.read_csv("SalesData.csv")[["GoodName", "StrFactDate", "SaleAmount"]]
    csv = convert_df(df)
    st.download_button("Sample Data", csv, "SampleData.csv","text/csv",
    key='download-csv')

