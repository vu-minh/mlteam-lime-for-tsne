# plot tsne embedding of country dataset with country name as tooltip

import joblib
import plotly.express as px
import pandas as pd
from utils import load_tabular_dataset

dataset_name = "country"
perplexity = 10
seed = 42
data_dir = f"./var/{dataset_name}"
embedding_name = f"{data_dir}/Y-perp{perplexity}-seed{seed}.Z"

X, labels, feature_names = load_tabular_dataset(dataset_name, standardize=True)
Y = joblib.load(embedding_name)

data_dict = dict(x=Y[:, 0].tolist(), y=Y[:, 1].tolist(), country_name=labels)
df = pd.DataFrame.from_dict(data_dict)

print(df.describe())

print(df.head())

fig = px.scatter(df, x="x", y="y", hover_name="country_name", width=800, height=800)
fig.show()
