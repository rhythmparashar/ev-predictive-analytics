#!/usr/bin/env python3

"""
Minimal SOC Model Analysis

Clean and stable ML diagnostics.

Outputs:

metrics.json

prediction_vs_actual.png
residual_hist.png

train_vs_val_residual_hist.png
train_vs_val_prediction.png
mae_by_soc_bucket.png
feature_drift_soc_pct.png

shap_summary.png
shap_bar.png
"""

from pathlib import Path
from datetime import datetime,timedelta
import argparse
import json
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.metrics import mean_absolute_error,mean_squared_error

import shap


# ------------------------
# Constants
# ------------------------

TARGET="y_soc_t_plus_300s"
LABEL="label_available"

META_COLS=[
"timestamp",
"vehicle_id",
"trip_id",
"quality_flag",
LABEL,
TARGET
]


# ------------------------
# Paths
# ------------------------

ROOT=Path(__file__).resolve().parents[1]

GOLD_DIR=ROOT/"data/gold/window_features"

REPORT_DIR=ROOT/"data/reports/model_analysis"


# ------------------------
# Utilities
# ------------------------

def daterange(start,end):

    d=start

    while d<=end:

        yield d

        d+=timedelta(days=1)


def load_gold(start,end,vehicle):

    dfs=[]

    for d in daterange(start,end):

        base=GOLD_DIR/f"dt={d}"

        if not base.exists():
            continue

        path=base/f"vehicle_id={vehicle}"

        if not path.exists():
            continue

        files=list(path.glob("*.parquet"))

        for f in files:
            dfs.append(pd.read_parquet(f))

    if not dfs:
        raise RuntimeError("No data")

    df=pd.concat(dfs)

    df["timestamp"]=pd.to_datetime(df["timestamp"])

    return df


def dataset(df):

    df=df[df[LABEL]==1]

    X=df.drop(columns=[c for c in META_COLS if c in df])

    y=df[TARGET]

    X=X.fillna(0)

    return X,y,df


def align(X,booster):

    X=X.reindex(columns=booster.feature_names)

    return X.fillna(0)


def metrics(y,p):

    return{

        "mae":
        float(mean_absolute_error(y,p)),

        "rmse":
        float(np.sqrt(
            mean_squared_error(y,p)
        ))
    }


# ------------------------
# Plots
# ------------------------

def plot_prediction(y,p,out):

    plt.figure()

    plt.scatter(y,p,s=3)

    plt.xlabel("Actual SOC")

    plt.ylabel("Predicted SOC")

    plt.savefig(out,dpi=150)

    plt.close()


def plot_residual(res,out):

    plt.figure()

    plt.hist(res,bins=80)

    plt.xlabel("Residual")

    plt.savefig(out,dpi=150)

    plt.close()


def plot_train_val_residual(tr,val,out):

    plt.figure()

    plt.hist(tr,bins=80,alpha=0.5,label="train")

    plt.hist(val,bins=80,alpha=0.5,label="val")

    plt.legend()

    plt.savefig(out,dpi=150)

    plt.close()


def plot_train_val_prediction(ytr,ptr,yv,pv,out):

    plt.figure()

    plt.scatter(ytr,ptr,s=3,label="train")

    plt.scatter(yv,pv,s=3,label="val")

    plt.legend()

    plt.savefig(out,dpi=150)

    plt.close()


def plot_soc_bucket(y,res,out):

    df=pd.DataFrame()

    df["y"]=y

    df["err"]=np.abs(res)

    df["bucket"]=pd.cut(
        df.y,
        bins=[0,20,40,60,80,100]
    )

    g=df.groupby("bucket")["err"].mean()

    g.plot(kind="bar")

    plt.ylabel("MAE")

    plt.savefig(out,dpi=150)

    plt.close()


def plot_soc_drift(tr,val,out):

    plt.figure()

    plt.hist(tr,bins=50,alpha=0.5,label="train")

    plt.hist(val,bins=50,alpha=0.5,label="val")

    plt.legend()

    plt.savefig(out,dpi=150)

    plt.close()


# ------------------------
# SHAP
# ------------------------

def run_shap(booster, X, out_dir, n):

    n = min(n, len(X), 200)

    print("Running SHAP on", n, "rows")

    Xs = X.sample(n, random_state=1)

    dmat = xgb.DMatrix(Xs)

    shap_values = booster.predict(
        dmat,
        pred_contribs=True
    )

    # Remove bias column (last column)
    shap_values = shap_values[:, :-1]

    shap.summary_plot(
        shap_values,
        Xs,
        show=False
    )

    plt.savefig(
        out_dir / "shap_summary.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    shap.summary_plot(
        shap_values,
        Xs,
        plot_type="bar",
        show=False
    )

    plt.savefig(
        out_dir / "shap_bar.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("SHAP complete")


# ------------------------
# Main
# ------------------------

def main():

    p=argparse.ArgumentParser()

    p.add_argument("--dt",required=True)

    p.add_argument("--vehicle",required=True)

    p.add_argument("--model",required=True)

    p.add_argument("--train-start")

    p.add_argument("--train-end")

    p.add_argument("--shap",action="store_true")

    args=p.parse_args()

    val_date=datetime.strptime(
    args.dt,"%Y-%m-%d").date()

    train_start=datetime.strptime(
    args.train_start,"%Y-%m-%d").date()

    train_end=datetime.strptime(
    args.train_end,"%Y-%m-%d").date()

    out=REPORT_DIR/f"dt={args.dt}/vehicle_id={args.vehicle}"

    out.mkdir(parents=True,exist_ok=True)


    booster=xgb.Booster()

    booster.load_model(args.model)


    # TRAIN

    df_tr=load_gold(
    train_start,
    train_end,
    args.vehicle)

    Xtr,ytr,raw_tr=dataset(df_tr)

    Xtr=align(Xtr,booster)

    ptr=booster.predict(
    xgb.DMatrix(Xtr))

    res_tr=ptr-ytr


    # VAL

    df_val=load_gold(
    val_date,
    val_date,
    args.vehicle)

    Xv,yv,raw_val=dataset(df_val)

    Xv=align(Xv,booster)

    pv=booster.predict(
    xgb.DMatrix(Xv))

    res_val=pv-yv


    # METRICS

    report={

    "train":metrics(ytr,ptr),

    "val":metrics(yv,pv)

    }

    with open(out/"metrics.json","w") as f:

        json.dump(report,f,indent=2)


    # PLOTS

    plot_prediction(yv,pv,
    out/"prediction_vs_actual.png")

    plot_residual(res_val,
    out/"residual_hist.png")

    plot_train_val_residual(
    res_tr,
    res_val,
    out/"train_vs_val_residual_hist.png")

    plot_train_val_prediction(
    ytr,ptr,
    yv,pv,
    out/"train_vs_val_prediction.png")

    plot_soc_bucket(
    yv,
    res_val,
    out/"mae_by_soc_bucket.png")

    plot_soc_drift(
    raw_tr.soc_pct,
    raw_val.soc_pct,
    out/"feature_drift_soc_pct.png")


    gc.collect()


    # SHAP

    if args.shap:

        run_shap(
        booster,
        Xv,
        out,
        200)


    print("Report:",out)



if __name__=="__main__":

    main()