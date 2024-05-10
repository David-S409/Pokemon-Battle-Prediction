from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt
from data_parse import model, X_train

# Get the first tree from the Random Forest model
tree = model.estimators_[0]

# Export the tree to a DOT file
dot_data = export_graphviz(
    tree,
    out_file=None,
    feature_names=X_train.columns,  # Use the feature names from X_train
    class_names=model.classes_,  # Use the class names from the model
    filled=True,
    rounded=True,
    special_characters=True,
)

# Visualize the DOT file using Graphviz
graph = graphviz.Source(dot_data)

# Display the tree diagram
plt.figure(figsize=(10, 10))
graph.render("pokemon_tree", format="png", cleanup=True)
plt.imshow(plt.imread("pokemon_tree.png"))
plt.axis("off")
plt.show()
