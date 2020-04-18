# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import re

from tqdm import notebook as tqdm
import attr
import pandas as pd
import numpy as np
import seaborn as sns
import networkx as nx
import geopy

import umap
import hdbscan
from matplotlib import pyplot as plt
from community import community_louvain

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from IPython.display import display

# %% [markdown]
# # Load data

# %%
df = pd.read_excel("data/FR-Questionnaire_francais.xlsx")

# %%
df.head(3)

# %%
raw_lifestyle_df = df.drop(
    columns=[
        "Avez-vous l'impression que les gens autour de vous respectent les mesures politiques prises ?",
        "Quels sont les sentiments que vous avez éprouvés depuis le début du confinement ?",
        "Que savez-vous sur le coronavirus ?",
        "Horodateur",
        "Avez-vous accès aux ressources suivantes ?",
        "Utilisez-vous un masque et/ou des gants lorsque vous sortez ?",
        "Quand avez-vous passé votre dernier examen de santé ?",
        "Avez-vous passé un test pour le coronavirus au cours des derniers mois ?",
        "Savez-vous que, pendant la durée du confinement, la meilleure façon de recevoir une aide médicale est d'appeler votre médecin, de travailler avec lui via les médias sociaux ou par appel vidéo ?",
        "Considérez-vous que vous êtes en bonne santé en ce moment ?",
    ]
)

# %% [markdown]
# # Process dataset

# %% [markdown]
# Translation utility for potentially translating the strings to a different language.

# %%
translation_strings = {}


def _(s):
    trans = translation_strings.get(s)
    if trans is None:
        return s
    else:
        return trans


# %% [markdown]
# Definitions of feature types present in the survey

# %%
def process_col(s):
    return s.lower().replace(" ", "_")


@attr.s
class Feature:
    name = attr.ib()


@attr.s
class CategoricalFeature(Feature):
    astype = attr.ib(default=float)

    def encode(self, df):
        drop_first = len(df[self.name].dropna().unique()) <= 2
        dummies = pd.get_dummies(
            df[self.name], drop_first=drop_first, prefix=self.name, prefix_sep="-"
        )
        return dummies.astype(self.astype).rename(columns=process_col)


@attr.s
class OrdinalFeature(Feature):
    """
    Categorical feature with order.
    """

    levels = attr.ib(default=None)
    astype = attr.ib(default=float)
    normalize = attr.ib(default=True)

    def encode(self, df):
        renaming_dict = {
            name: ord for name, ord in zip(self.levels, range(len(self.levels)))
        }
        col = df[self.name].astype(str).replace(renaming_dict).astype(self.astype)
        if self.normalize:
            col = (col - col.mean()) / col.std()
        return col


@attr.s
class MultiCatFeature(Feature):
    """
    Multiple categories feature, written as a comma-separated string.
    """

    sep = attr.ib(default=", ")
    astype = attr.ib(default=float)

    def encode(self, df):
        dummies = df[self.name].str.get_dummies(sep=self.sep).astype(self.astype)
        return dummies.rename(columns=process_col).rename(
            columns=lambda x: f"{self.name}-{x}"
        )


@attr.s
class NumFeature(Feature):
    """
    Numeric feature
    """

    max_val = attr.ib(default=np.inf)
    astype = attr.ib(default=float)
    normalize = attr.ib(default=True)

    def encode(self, df):
        def to_num(x):
            num_str = re.sub("[^.0-9]", "", str(x))
            if len(num_str) > 0:
                val = float(num_str)
                if val <= self.max_val:
                    return val

        col = df[self.name].apply(to_num).astype(self.astype)
        if self.normalize:
            col = (col - col.mean()) / col.std()
        return col


# %% [markdown]
# Features specifications: dictionary that maps raw column names to a feature definition object.

# %%
features = {
    _("Comment restez-vous informé·e de la pandémie COVID-19 ?"): MultiCatFeature(
        name="information_channel", sep=", "
    ),
    _("Comment jugez-vous l'impact sur votre vie quotidienne ?"): OrdinalFeature(
        name="impact_on_life",
        levels=[_("Pas d'impact"), _("Léger"), _("Modéré"), _("Significatif")],
    ),
    _("Quels endroits avez-vous visités la semaine dernière ?"): MultiCatFeature(
        name="places_visited", sep=", "
    ),
    _(
        "Combien de fois êtes-vous sorti de chez vous la semaine dernière ?"
    ): OrdinalFeature(
        name="number_exits",
        levels=[
            _("Aucune"),
            _("Entre 1 et 3 fois"),
            _("4 à 7 fois"),
            _("Plus de 7 fois"),
        ],
    ),
    _("Quel type de transport possédez-vous ?"): CategoricalFeature(name="transport"),
    _(
        "Au cours de la semaine dernière, avec combien de personnes avez-vous "
        "été en contact physique étroit (moins de 2 mètres, plus de 10 min), en dehors de votre foyer ?"
    ): NumFeature(name="num_close_contact", max_val=500),
    _("Avez-vous voyagé au cours du dernier mois ?"): CategoricalFeature(
        name="recent_travel"
    ),
    _(
        "Êtes-vous bénévole dans une activité pour aider à lutter contre la pandémie ?"
    ): CategoricalFeature(name="volunteer"),
    _(
        "Partagez-vous l'une de ces installations avec des voisins ou des étrangers ?"
    ): MultiCatFeature(name="shared_installation", sep=", "),
    _("Comment travaillez-vous ou étudiez-vous maintenant ?"): CategoricalFeature(
        name="work_mode"
    ),
    _(
        "À quelle distance se trouve votre lieu d'approvisionnement alimentaire le plus proche ?"
    ): OrdinalFeature(
        name="distance_to_food_shop",
        levels=["Moins de 15 minutes", "15-30 minutes", "Plus de 30 minutes"],
    ),
    _("Comment vous procurez-vous de la nourriture ?"): MultiCatFeature(
        name="food_shop_type", sep=", "
    ),
    _("À quelle fréquence achetez-vous de la nourriture ?"): OrdinalFeature(
        name="shopping_frequency",
        levels=[
            _("Chaque jour"),
            _("Tous les 3 jours environ"),
            _("Chaque semaine"),
            _("Toutes les deux semaines"),
            _("Moins souvent"),
        ],
    ),
    _(
        "Depuis le début du confinement vous mangez en quantité : inférieure, identique, "
        "supérieure à d'habitude ?"
    ): OrdinalFeature(
        name="food_habit_change",
        levels=[_("Inférieure"), _("Identique"), _("Supérieure")],
    ),
    _(
        "Avec qui êtes-vous le plus en contact (virtuellement, en dehors de votre foyer) ?"
    ): CategoricalFeature(name="most_frequent_contact"),
    _(
        "Comment communiquez-vous à distance avec votre famille et vos amis ?"
    ): MultiCatFeature(name="relatives_contact_mean", sep=", "),
    _("Comment communiquez-vous à distance avec vos collègues ?"): MultiCatFeature(
        name="work_contact_mean", sep=", "
    ),
    _(
        "Depuis le début du confinement, à quelle fréquence communiquez-vous à distance avec "
        "vos amis et votre famille en dehors du foyer ?"
    ): OrdinalFeature(
        name="relatives_contact_frequency",
        levels=[
            _("Plusieurs fois par jour"),
            _("Tous les jours"),
            _("Tous les 3 jours environ"),
            _("Toutes les semaines"),
            _("Toutes les deux semaines"),
            _("Très rarement"),
            _("Pas du tout"),
        ],
    ),
    _(
        "Durant la semaine dernière, avec combien d'ami·e·s ou de proches avez-vous communiqué à distance ?"
    ): OrdinalFeature(
        name="num_relatives_contact",
        levels=[
            _("Aucun"),
            _("1-3"),
            _("4-6"),
            _("7-10"),
            _("11-20"),
            _("Plus de 20"),
        ],
    ),
    _("Quel est votre âge ?"): NumFeature(name="age"),
    _("Quel est votre genre ?"): CategoricalFeature(name="gender"),
    _("Quelle est votre profession ?"): CategoricalFeature(name="profession"),
    _(
        "Comment jugez-vous votre niveau d’exposition au coronavirus au travail ?"
    ): OrdinalFeature(name="work_exposure", levels=["Faible", "Moyen", "Élevé"]),
    _(
        "Vous compris, combien de personnes vivent dans le même ménage que vous ?"
    ): NumFeature(name="household_size"),
    _("Partagez-vous votre chambre ?"): CategoricalFeature(name="shared_room"),
}

# %%
dfs = []

# Create a dataframe with short column names for convenience
feature_names = [f.name for f in features.values()]
lifestyle_df = raw_lifestyle_df.rename(
    columns={old: new for old, new in zip(features.keys(), feature_names)}
)

# Encode features
for feature in features.values():
    feature_df = feature.encode(lifestyle_df)
    dfs.append(feature_df)

processed_data = pd.concat(dfs, axis=1)
processed_data

# %% [markdown]
# Sanity check. Looking at all the unique feature values to see if all of them make sense.

# %%
for col in processed_data.columns:
    print(col, processed_data[col].unique())

# %% [markdown]
# We use Nominatim to get canonical locations (in lat, lon) for the locations that respondents have typed in free form. Set `compute_locations` to `True` to run this computation.

# %%
compute_locations = False

if compute_locations:
    locations, latitude, longitude = [], [], []
    for i, row in tqdm.tqdm(
        list(
            lifestyle_df[
                ["Dans quel ville vivez-vous ?", "Dans quel pays vivez-vous ?"]
            ].iterrows()
        )
    ):
        query = ""
        if type(row["Dans quel ville vivez-vous ?"]) is str:
            query += str(row["Dans quel ville vivez-vous ?"]) + ", "
        if type(row["Dans quel pays vivez-vous ?"]) is str:
            query += str(row["Dans quel pays vivez-vous ?"]) + ", "

        geocoder = geopy.geocoders.Nominatim(user_agent="pdm", timeout=10)
        query_result = geocoder.geocode(query)
        address = query_result.address if query_result is not None else None
        locations.append(address)
        lat = query_result.latitude if query_result is not None else None
        latitude.append(lat)
        lon = query_result.latitude if query_result is not None else None
        longitude.append(lon)

    df_location = pd.DataFrame(
        {"location": locations, "lat": latitude, "lon": longitude}
    )
    df_location.to_csv("data/df_location.csv")

# %% [markdown]
# ## Build social network and cluster communities

# %% [markdown]
# Empirically, we want to remove these two columns, because only one person self-reported themselves with the "other" gender, and this happened to skew the clustering quite a bit.

# %%
clean_data = processed_data.drop(columns=["gender-autre", "gender-m"],)

# %% [markdown]
# First step of the Louvain clustering algorithm is to create a feature similarity matrix. We use the number of exactly matching feature values as the similarity score here.

# %%

matrix = np.zeros((len(clean_data), len(clean_data)))

for i, row_i in tqdm.tqdm(list(clean_data.iterrows())):
    for j, row_j in clean_data.iterrows():
        if j > i:
            score = np.sum((row_i.values == row_j.values))
            matrix[i][j] = score
            matrix[j][i] = score

# %%
np.min(matrix[matrix > 0]), np.max(matrix), np.mean(matrix[matrix > 0]), np.median(
    matrix[matrix > 0]
)

# %%
# Normalisation
processed_matrix = (matrix >= np.percentile(matrix[matrix > 0], 95)).astype("bool")

# %%
G = nx.from_numpy_matrix(processed_matrix)

# %%
print(nx.info(G))

# %%
nx.write_gexf(G, "social_graph.gexf")

# %%
partition = community_louvain.best_partition(G, random_state=0)

# %%
communities_data = clean_data.copy()
communities_data["community"] = pd.Series(partition)

# %% [markdown]
# This is the list of communities obtained using the Louvain clustering with their respective member counts.

# %%
list_communities = communities_data["community"].value_counts()
list_communities

# %%
community_annotated_raw_df = lifestyle_df.copy().assign(community=pd.Series(partition))

# %% [markdown]
# We decide to only take the communities that have more than 5 people, and rename them as 1, 2, 3, ... in the order of increasing size. All communities smaller or equal than the threshold get lumped into an "outlier" community -1.

# %%
COMMUNITY_SIZE_THRESHOLD = 5

community_renaming_dict = {}
counter = 1
for i, count in list_communities.iteritems():
    if count <= COMMUNITY_SIZE_THRESHOLD:
        community_renaming_dict[i] = -1
    else:
        community_renaming_dict[i] = counter
        counter += 1

# %%
community_annotated_raw_df.community.replace(community_renaming_dict, inplace=True)

# %%
community_annotated_raw_df.community.value_counts()

# %%
community_annotated_raw_df.to_excel("out/FR_Questionnaire_communities.xlsx")

# %% [markdown]
# ## Analysis of the communities

# %% [markdown]
# ### Outliers

# %%
for isolated in list_communities[list_communities == 1].index:
    individual = df[communities_data.community == isolated]
    # NOTE: We add +2 to the index so that to match the row number in spreadsheet viewing software.
    print(f"Individual {individual.index[0] + 2}")

# %% [markdown]
# ### Main communities

# %%
df_main_communities = (
    communities_data.replace(community_renaming_dict).query("community != 1").dropna()
)

# %% [markdown]
# We build an interpretable classifier that aims to predict the community based on the survey responses.

# %%
X, y = shuffle(
    df_main_communities.drop(columns="community"),
    df_main_communities.community,
    random_state=0,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# %%
baseline = max(df_main_communities.community.value_counts() / len(df_main_communities))
print(f"Random baseline: {baseline}")

# %%
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

# %%
clf = LogisticRegressionCV()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
best_param = clf.C_.mean()

# %%
X, y = df_main_communities.drop(columns="community"), df_main_communities["community"]

clf = LinearDiscriminantAnalysis()
clf.fit_transform(X, y)
persona = clf.decision_function(df_main_communities.drop(columns="community"))

# %%
X, y = df_main_communities.drop(columns="community"), df_main_communities["community"]

clf = LogisticRegression(C=best_param)
clf.fit(X, y)
persona = clf.decision_function(df_main_communities.drop(columns="community"))

# %%
decisions = clf.decision_function(X)
probabilities = np.exp(decisions) / np.sum(np.exp(decisions))

# %% [markdown]
# Display most important features and typical representatives

# %%
TOP_K = 10

for i, community in enumerate(clf.classes_):
    # Find the "typical representative" of the community: the one with the highest score.
    # NOTE: We add +2 to the index so that to match the row number in spreadsheet viewing software.
    idx = X.iloc[np.argmax(np.abs(probabilities[:, i]))].name + 2
    if i == -1:
        print(f"Typical outlier (-1): {idx}")
    else:
        print(f"Typical individual for community {community}: {idx}")

    # Show top k features for this community.
    for top_question in reversed(np.argsort(np.abs(clf.coef_[i, :]))[-TOP_K:]):
        print("   Question:", X.columns[top_question])
        print("   Effect size:", clf.coef_[i, :][top_question])
        print()

# %% [markdown]
# Make a spreadsheet with community demographics.

# %%
data = df_communities.copy()
data.community.replace(community_renaming_dict, inplace=True)
data.city = lifestyle_df[_("Dans quel ville vivez-vous ?")]
data.country = lifestyle_df["Dans quel pays vivez-vous ?"]

for i, community in enumerate(list(clf.classes_) + ["Total"]):
    if community != "Total":
        community_data = data.query(f"community == {community}")
    else:
        community_data = data
    age = community_data.age * lifestyle_df.age.std() + lifestyle_df.age.mean()
    demographics = {
        "community": community,
        "age": age.mean(),
        "age_std": age.std(),
        "gender_prop_woman": community_data["gender-f"].mean(),
    }
    if i == 0:
        print(",".join(demographics.keys()))
    print(",".join([str(v) for v in demographics.values()]))
