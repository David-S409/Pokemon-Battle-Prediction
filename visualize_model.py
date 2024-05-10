import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
)
import seaborn as sns

from util import get_feature_list, pretty_print

from data_parse import X_test, y_test, y_pred

features = get_feature_list()
model = joblib.load("./models/battle_model.pkl")

feature_importances = model.feature_importances_

# Create a DataFrame to display feature importances
importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importances})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Print or display the sorted importance DataFrame
pretty_print("Important", importance_df)
