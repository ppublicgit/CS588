import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris


### Load in Iris data
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
convert_species = np.vectorize(lambda x : "setosa" if x==0 else ("versicolor" if x==1 else "virginica"))
iris_df["species"] = convert_species(iris.target)

### Get Correlation and plot heatmap
corr = iris_df.corr()
print(corr)
plt.figure()
sns.heatmap(corr, annot=True)
plt.show(block=False)

### plot pair plot of iris data
sns.pairplot(iris_df, hue="species", diag_kind="hist")
plt.show()
