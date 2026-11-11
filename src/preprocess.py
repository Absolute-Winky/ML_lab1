import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUM = ["year", "mileage", "tax", "mpg", "engineSize", "car_age", "mpy"]
CAT = ["model", "transmission", "fuelType"]

def load_clean(path="data/merc.csv"):
    df = pd.read_csv(path)
    df = df[df["engineSize"] != 0]
    df = df[df["mpg"] <= 150]
    df = df[df["year"] >= 2000]
    df = df[df["price"] <= 100000]
    freq = df["model"].value_counts()
    rare = freq[freq < 50].index
    df.loc[df["model"].isin(rare), "model"] = "Other"
    df["car_age"] = 2021 - df["year"]
    df["mpy"] = df["mileage"] / df["car_age"].clip(lower=1)
    return df.reset_index(drop=True)

def train_val_test(df, test_size=0.2, val_size=0.25, random_state=42):
    bins = pd.qcut(df["price"], 5, labels=False, duplicates="drop")
    tr, te = train_test_split(df, test_size=test_size, random_state=random_state, stratify=bins)
    bins_tr = pd.qcut(tr["price"], 5, labels=False, duplicates="drop")
    tra, va = train_test_split(tr, test_size=val_size, random_state=random_state, stratify=bins_tr)
    return tra.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)

def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])

def split_xy(df):
    return df[NUM + CAT], df["price"]
