import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your preprocessed dataset (preprocessed_df from the previous example)
df = preprocessed_df.copy()

# Feature Selection and Engineering
# Assuming you've already selected relevant features including 'type1', 'type2', and 'speed'
selected_features = ["type1", "type2", "speed"]  # Update with your selected features
X = df[selected_features]
y = df["name"]  # Target variable (assuming 'name' is the Pokémon's name)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train a Random Forest Classifier
