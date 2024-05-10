## Data Sets
* Pokemon stat data
* Battle outcomes

## Features
1. Type
2. HP
3. AttackScore (max(Attack, Sp_Attack) * speed)
4. AvgDefense ((Defense + Sp. Defense) / 2)
5. AttackDefenseRatio (AttackScore / AvgDefense)




## Original Features
feature_list = [
    "HP_1",
    "Attack_1",
    "Defense_1",
    "Sp_Attack_1",
    "Sp_Defense_1",
    "Speed_1",
    "Generation_1",
    "Legendary_1",
    "TypeAdvantages_1",
    "TypeDisadvantages_1",
    "AttackScore_1",
    "AvgDefense_1",
    "AttackDefenseRatio_1",
    "HP_2",
    "Attack_2",
    "Defense_2",
    "Sp_Attack_2",
    "Sp_Defense_2",
    "Speed_2",
    "Generation_2",
    "Legendary_2",
    "TypeAdvantages_2",
    "TypeDisadvantages_2",
    "AttackScore_2",
    "AvgDefense_2",
    "AttackDefenseRatio_2",
]

* 88% Accurate


## Classification Scores

|            | Precision | Recall | F1-Score | Support  |
|------------|-----------|--------|----------|----------|
| Accuracy   |           |        | 0.88     | 10000    |
| Macro Avg  | 0.81      | 0.80   | 0.79     | 10000    |
| Weight Avg | 0.88      | 0.88   | 0.97     | 10000    |


## Validation
Cross-Validation Scores: [0.8892 0.8878 0.8899 0.8859 0.8925]
Mean CV Accuracy: 0.88906