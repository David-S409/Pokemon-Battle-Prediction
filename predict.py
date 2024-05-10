import json
from webbrowser import get
import joblib
from matplotlib.pyplot import sca
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from util import (
    calculate_type_advantages,
    calculate_type_disadvantages,
    pretty_print,
    get_feature_list,
    scale_data,
    scaling_columns,
)

# Load Pokemon data
pokemon_data = json.load(open("./data/pokemon.json"))


def get_pokemon_data(id):
    if id < 1 or id > len(pokemon_data):
        return None
    return pokemon_data[id - 1]


def calculate_features(pokemon_array):
    output_dt = pd.DataFrame()
    if len(pokemon_array) != 2:
        return None

    for i, pokemon in enumerate(pokemon_array):
        if pokemon is None:
            return None

        # Convert all strings to int where applicable
        for key in pokemon:
            if key not in ["Type1", "Type2", "Name", "Legendary"]:
                pokemon[key] = int(pokemon[key])

        higher_attack = max(pokemon["Attack"], pokemon["Sp_Attack"])
        attack_score = higher_attack * pokemon["Speed"]
        average_defense = (pokemon["Defense"] + pokemon["Sp_Defense"]) / 2
        attack_defense_ratio = attack_score / average_defense

        pokemon_metric = {
            "Pokemon_ID": pokemon["Pokemon_ID"],
            "Type1": pokemon["Type1"],
            "Type2": pokemon["Type2"],
            "HP": pokemon["HP"],
            "Attack": pokemon["Attack"],
            "Defense": pokemon["Defense"],
            "Sp_Attack": pokemon["Sp_Attack"],
            "Sp_Defense": pokemon["Sp_Defense"],
            "Speed": pokemon["Speed"],
            "Generation": pokemon["Generation"],
            "Legendary": pokemon["Legendary"] == "True",
            "AttackScore": attack_score,
            "AvgDefense": average_defense,
            "AttackDefenseRatio": attack_defense_ratio,
        }

        input_df = pd.DataFrame([pokemon_metric])

        if pokemon["Type2"] == "":
            input_df["Type2"] = "None"

        input_df["TypeAdvantages"] = input_df.apply(calculate_type_advantages, axis=1)
        input_df["TypeDisadvantages"] = input_df.apply(
            calculate_type_disadvantages, axis=1
        )

        input_df = input_df.drop(["Type1", "Type2"], axis=1)

        input_df = input_df.rename(
            columns={
                "Pokemon_ID": "Pokemon_ID_" + str(i + 1),
                "HP": "HP_" + str(i + 1),
                "Attack": "Attack_" + str(i + 1),
                "Defense": "Defense_" + str(i + 1),
                "Sp_Attack": "Sp_Attack_" + str(i + 1),
                "Sp_Defense": "Sp_Defense_" + str(i + 1),
                "Speed": "Speed_" + str(i + 1),
                "Generation": "Generation_" + str(i + 1),
                "Legendary": "Legendary_" + str(i + 1),
                "TypeAdvantages": "TypeAdvantages_" + str(i + 1),
                "TypeDisadvantages": "TypeDisadvantages_" + str(i + 1),
                "AttackScore": "AttackScore_" + str(i + 1),
                "AvgDefense": "AvgDefense_" + str(i + 1),
                "AttackDefenseRatio": "AttackDefenseRatio_" + str(i + 1),
            }
        )

        output_dt = pd.concat([output_dt, input_df], axis=1)

    return output_dt


def predict_outcomes(pokemon1ID: int, pokemon2ID: int):

    pokemon1 = get_pokemon_data(pokemon1ID)
    pokemon2 = get_pokemon_data(pokemon2ID)

    pokemon_features = calculate_features([pokemon1, pokemon2])
    model = joblib.load("./models/battle_model.pkl")

    input_df = pd.DataFrame([pokemon_features.iloc[0]])

    # Reorder the columns to match the model's feature order
    feature_list = get_feature_list()
    input_df = input_df[feature_list]

    predicted_class = model.predict(input_df)[0]
    pretty_print("Predicted Class Label:", predicted_class)

    # Battle Prediction
    battle_info = {
        "Pokemon1": pokemon1["Name"],
        "Pokemon2": pokemon2["Name"],
        "Prediction": get_pokemon_data(predicted_class)["Name"],
    }

    pretty_print("Battle Info:", battle_info)
