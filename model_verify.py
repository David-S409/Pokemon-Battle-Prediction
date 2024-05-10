import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score


# Calculate accuracy
def calculate_accuracy(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")


# Print classification report
def print_classiofication_report(y_test, y_pred):
    report = classification_report(y_test, y_pred)
    print(report)


# Calculate feature importance
def calculate_feature_important(model, features):
    feature_importance = model.feature_importances_
    feature_importance_df = pd.DataFrame(
        {"Feature": features, "Importance": feature_importance}
    )
    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    )
    print(feature_importance_df)


# Cross-validation
def calculate_validation_matrix(model, X, y):
    cv_scores = cross_val_score(model, X, y, cv=5)
    print("Cross-Validation Scores:", cv_scores)
    print("Mean CV Accuracy:", cv_scores.mean())
