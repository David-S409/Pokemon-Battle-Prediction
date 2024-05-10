from operator import ne
from typing import final
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import plot_tree
from util import (
    calculate_type_advantages,
    calculate_type_disadvantages,
    pretty_print,
    scale_data,
    scaling_columns,
    feature_list,
    new_all_types,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import seaborn as sns

from model_verify import (
    calculate_validation_matrix,
    classification_report,
    cross_val_score,
    print_classiofication_report,
    calculate_accuracy,
)

# Define type effectiveness


def calculate_attack_score(row):
    # Choose the higher value between Attack and Sp. Atk
    higher_attack = max(row["Attack"], row["Sp_Attack"])
    # Calculate the attack score by multiplying the higher attack value with Speed
    attack_score = higher_attack * row["Speed"]
    return attack_score


def calculate_attack_defense_ratio(row):
    # Calculate the attack-defense ratio by dividing Attack Score with AvgDefense
    attack_defense_ratio = row["AttackScore"] / row["AvgDefense"]
    return attack_defense_ratio


def calculate_type_math(row):
    # Get p1 type scalar
    p1_attack_scalar = new_all_types[row["Type1_1"]][row["Type1_2"]]
    # Get p2 type scalar
    p2_attack_scalar = new_all_types[row["Type1_2"]][row["Type1_1"]]

    # multiply the scalars
    row["AttackScore_1"] = row["AttackScore_1"] * p1_attack_scalar
    row["AvgDefense_2"] = row["AvgDefense_2"] * p2_attack_scalar


# Load your data
poke_data = pd.read_csv("./data/pokemon.csv")
poke_data["Legendary"] = poke_data["Legendary"].astype(int)
battle_data = pd.read_csv("./data/combats.csv")
poke_data.Type2 = poke_data.Type2.fillna("None")

# Drop legendary and generation columns
poke_data = poke_data.drop(["Generation", "Legendary"], axis=1)

poke_data["TypeAdvantages"] = poke_data.apply(calculate_type_advantages, axis=1)
poke_data["TypeDisadvantages"] = poke_data.apply(calculate_type_disadvantages, axis=1)


poke_data["AvgDefense"] = (poke_data["Defense"] + poke_data["Sp_Defense"]) / 2
poke_data["AttackScore"] = poke_data.apply(calculate_attack_score, axis=1)
poke_data["AttackDefenseRatio"] = poke_data.apply(
    calculate_attack_defense_ratio, axis=1
)

merged_data_first = pd.merge(
    battle_data,
    poke_data,
    left_on="First_pokemon",
    right_on="Pokemon_ID",
    how="left",
)

merged_data_second = pd.merge(
    merged_data_first,
    poke_data,
    left_on="Second_pokemon",
    right_on="Pokemon_ID",
    how="left",
    suffixes=("_1", "_2"),
)

pokemon1_types = merged_data_second[["Type1_1", "Type2_1"]]
pokemon2_types = merged_data_second[["Type1_2", "Type2_2"]]

merged_data_second.apply(calculate_type_math, axis=1)

# drop the type columns
poke_data = poke_data.drop(["Type1", "Type2", "Name"], axis=1)

final_data = merged_data_second
final_data["Speed_Ratio"] = final_data["Speed_1"] / final_data["Speed_2"]


X = final_data[feature_list]  # features is the list of features you want to use
y = final_data["Winner"]  # Winner is your target variable

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Train the classifier
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# # Evaluate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")

# save the model
# joblib.dump(model, "./models/battle_model.pkl")

calculate_validation_matrix(model, X, y)
