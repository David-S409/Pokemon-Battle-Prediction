from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from util import pretty_print

# import pokemon as pk
poke_data = pd.read_csv("./data/pokemon.csv")

# add a trend line with a scatter plot
plt.figure(figsize=(10, 6))
sns.regplot(x="Attack", y="Defense", data=poke_data)


plt.title("Attack vs. Defense")
plt.xlabel("Attack")
plt.ylabel("Defense")
plt.show()
