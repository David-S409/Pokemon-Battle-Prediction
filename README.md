# Pokémon Battle Matchup Prediction

## Overview

This project aims to predict the outcomes of Pokémon battles using machine learning techniques. By analyzing Pokémon stats and battle results, the model predicts which Pokémon is more likely to win in a given matchup. The project includes data preprocessing, feature engineering, model training, and evaluation.

## Dataset

The project uses two datasets obtained from Kaggle:
- Pokémon Stats Dataset: Contains attributes such as HP, Attack, Defense, Type, etc.
- Pokémon Battle Outcomes Dataset: Records the results of battles between Pokémon.

## Implementation Approach

1. **Data Exploration and Preprocessing**:
   - Explored and integrated Pokémon stats and battle outcome datasets.
   - Handled missing data and standardized features.

2. **Feature Engineering**:
   - Created new features like AttackScore, AvgDefense, TypeAdvantages, etc.
   - Analyzed feature importance and removed less significant features.

3. **Model Selection and Training**:
   - Chose Random Forests model for prediction.
   - Trained the model using 80% of battle records and tested on 20%.

4. **Model Evaluation**:
   - Achieved an accuracy of 0.88 in predicting battle outcomes.
   - Validated model performance using cross-validation.

## Results

- Accuracy: 0.88
- Classification Scores:
  - Precision: 0.88 (Weighted Avg.)
  - Recall: 0.88 (Weighted Avg.)
  - F1-Score: 0.88 (Weighted Avg.)
- Mean CV Accuracy: 0.87744

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/David-S409/Pokemon-Battle-Prediction
    ```

2. Install dependencies:
    
    ```bash
    pip install -r requirements.txt
    ```

2. Build the ML model:

   ```bash
   python data_parse.py
    ```

3. Run the prediction script:

    ```bash
    python pokemon_model.py <Pokemon_ID1> <Pokemon_ID1>
    ```
    Replace `<Pokemon_ID1>` and `<Pokemon_ID2>` with the IDs of the two Pokémon you want to predict the battle outcome for.

    Pokemon IDs can be found in the `data/pokemon.csv` file.

## References

- Kaggle Pokémon Stats and battle Outcome Dataset: [https://www.kaggle.com/datasets/terminus7/pokemon-challenge/data](URL)

## Contributors

- [David Santana](https://github.com/David-S409)
- [Omar Ramirez](https://github.com/Ramirez0245)

## License

This project is licensed under the [MIT License](LICENSE).
