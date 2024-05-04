import random
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, ParameterGrid, KFold, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, precision_score, jaccard_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from itertools import combinations

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import pickle

def print_scores(X_train: list, y_train: list, X_test: list, y_test: list, target_models: list):
    for i in range(0, len(target_models)):

        target_model = target_models[i]

        y_pred_train = target_model.predict(X_train[i])
        y_pred_test = target_model.predict(X_test[i])

        # Accuracy score
        accuracy_train = accuracy_score(y_train[i], y_pred_train)
        accuracy_test = accuracy_score(y_test[i], y_pred_test)

        print(f"Accuracy score for train {i + 1}: {accuracy_train:.4f}")
        print(f"Accuracy score for test {i + 1}: {accuracy_test:.4f}")

        # Precision score
        precision_train = precision_score(y_train[i], y_pred_train)
        precision_test = precision_score(y_test[i], y_pred_test)

        print(f"Precision score for train {i + 1}: {precision_train:.4f}")
        print(f"Precision score for test {i + 1}: {precision_test:.4f}")

        # Recall score
        recall_train = recall_score(y_train[i], y_pred_train)
        recall_test = recall_score(y_test[i], y_pred_test)

        print(f"Recall score for train {i + 1}: {recall_train:.4f}")
        print(f"Recall score for test {i + 1}: {recall_test:.4f}")

        # F1 score
        f1_train = f1_score(y_train[i], y_pred_train)
        f1_test = f1_score(y_test[i], y_pred_test)

        print(f"F1 score for train {i + 1}: {f1_train:.4f}")
        print(f"F1 score for test {i + 1}: {f1_test:.4f}")

        print("\n")


def print_best_model(target_models: list):
    for i in range(0, len(target_models)):

        target_model = target_models[i]

        print("\n The best estimator:\n", target_model["model"].best_estimator_)
        print("\n The best estimator:\n", target_model["model"].best_params_)
        print("\n The best score:\n", target_model["model"].best_score_)
        print("\n")


def print_feature_importance(target_models: list):
    for i in range(0, len(target_models)):

        target_model = target_models[i]

        target_importance = pd.DataFrame(target_model["model"].best_estimator_.feature_importances_, 
                                        target_model["preprocessors"].get_feature_names_out()
                                        ).reset_index().sort_values(by = 0, ascending = False).head(25)
        
        plt.figure(figsize = (9, 6))
        sns.barplot(x = 0, y = "index", data = target_importance)
        plt.show()

def create_bins(X_test, y_test, target_model):
    probabilities = target_model.predict_proba(X_test)[:, 1]

    target_pred = pd.DataFrame({"pred": probabilities,
                                "actual": y_test})

    target_pred["bin"] = pd.qcut(target_pred.pred.rank(method = "first"), 10)
    ozet = pd.DataFrame()

    toplam = pd.DataFrame(target_pred.groupby("bin").count()["actual"]).reset_index()["actual"]
    good = pd.DataFrame(target_pred.groupby("bin")["actual"].sum()).reset_index()["actual"]

    ozet["bin"] = range(10, 0, -1)
    ozet["max_skor"] = pd.DataFrame(target_pred.groupby("bin").max()).reset_index()["pred"]
    ozet["toplam"] = toplam
    ozet["good"] = good
    ozet["bad"] = toplam - good
    ozet["good%"] = good / toplam

    print(ozet.sort_values(by = "bin"))

def multiply_columns(df, column_list):
    """
    Multiply columns in every two combinations and save as new columns in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the columns.
        column_list (list): List of column names to be multiplied.

    Returns:
        pd.DataFrame: A new DataFrame with additional columns containing the multiplications.
    """
    new_df = df.copy()

    for col1, col2 in combinations(column_list, 2):
        new_col_name = f"multi_{col1}_{col2}"
        new_df[new_col_name] = df[col1] * df[col2]

    return new_df

def create_target(dataframe: pd.DataFrame):
    for i in range(1, 10):
        dataframe[f"target_{i}"] = np.where(dataframe["target"].str.contains(pat = f"menu{i}", regex = True), 1, 0)

    return dataframe

def modify_columns(dataframe: pd.DataFrame):
    menu_list = ["first_menu", "second_menu", "third_menu"]
    n_seconds_list = ["n_seconds_1", "n_seconds_2", "n_seconds_1"]
    output_dataframes = []

    dataframe[menu_list] = dataframe['target'].str.split(',', expand=True)
    for i in range(0, 3):
        dataframe[menu_list[i]] = dataframe[menu_list[i]].str.strip()

    for i in range(0, 3):
        df = dataframe[["id", "month", n_seconds_list[i], menu_list[i]]]
        df.rename(columns = {n_seconds_list[i]: "n_seconds", menu_list[i]: "menu"}, inplace = True)
        output_dataframes.append(df)

    return output_dataframes[0], output_dataframes[1], output_dataframes[2]

def sample_df(dataframe:pd.DataFrame, target:str):
    sample_size = dataframe[target].value_counts().min()
    selected_df = pd.DataFrame()

    for label_value in dataframe[target].unique():
        label_samples = dataframe[dataframe[target] == label_value].sample(sample_size, random_state = 1)
        selected_df = pd.concat([selected_df, label_samples])

    return selected_df

def create_target_df(dataframe: pd.DataFrame, feature_flag: list, n_seconds_flag: list, categoric_flag: list, multi_flag: list):
    target_dataframes = []

    for i in range(1, 10):
        target_cols = ["id"] + feature_flag + n_seconds_flag + categoric_flag + multi_flag + [f"target_{i}"]
        df = dataframe.filter(items = target_cols)
        df.set_index("id", inplace = True)
        selected_df = sample_df(df, f"target_{i}")
        target_dataframes.append(selected_df)

    return target_dataframes

def show_counts(dataframe: pd.DataFrame, column: str):
    counts = dataframe[column].value_counts()
    
    return counts

def create_train_test(target_dataframes: list, target_list: list):
    assert len(target_dataframes) == len(target_list)
    input_target_pairs = []
    train_test_pairs = []

    for i in range(0, 9):
        X = target_dataframes[i].drop(target_list[i], axis = 1)
        y = target_dataframes[i][target_list[i]]
        input_target_pairs.append((X, y))
        train_test_pairs.append((train_test_split(X, y, test_size = 0.2, random_state = 31, stratify = y)))

    return input_target_pairs, train_test_pairs