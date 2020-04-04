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
import geopy

# %% [markdown]
# # Load data

# %%
df = pd.read_excel("data/FR-Questionnaire_francais.xlsx")

# %%
df.head(3)

# %% [markdown]
# # Refine dataset

# %%
translation_strings = {}


def _(s):
    trans = translation_strings.get(s)
    if trans is None:
        return s
    else:
        return trans


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
    levels = attr.ib(default=None)
    astype = attr.ib(default=float)

    def encode(self, df):
        renaming_dict = {
            name: ord for name, ord in zip(self.levels, range(len(self.levels)))
        }
        return df[self.name].astype(str).replace(renaming_dict).astype(self.astype)


@attr.s
class MultiCatFeature(Feature):
    sep = attr.ib(default=", ")
    astype = attr.ib(default=float)

    def encode(self, df):
        dummies = df[self.name].str.get_dummies(sep=self.sep).astype(self.astype)
        return dummies.rename(columns=process_col).rename(
            columns=lambda x: f"{self.name}-{x}"
        )


@attr.s
class NumFeature(Feature):
    max_val = attr.ib(default=np.inf)
    astype = attr.ib(default=float)
    
    def encode(self, df):
        def to_num(x):
            num_str = re.sub("[^.0-9]", "", str(x))
            if len(num_str) > 0:
                val = float(num_str)
                if val <= self.max_val:
                    return val

        return df[self.name].apply(to_num).astype(self.astype)


@attr.s
class CityFeature(Feature):
    def encode(self, df):
        pass


@attr.s
class CountryFeature(Feature):
    def encode(self, df):
        pass


# %%
lifestyle_df = df.drop(
    columns=[
        "Avez-vous l'impression que les gens autour de vous respectent les mesures politiques prises ?",
        "Quels sont les sentiments que vous avez éprouvés depuis le début du confinement ?",
        "Que savez-vous sur le coronavirus ?",
        "Horodateur",
        "Avez-vous accès aux ressources suivantes ?",
        "Utilisez-vous un masque et/ou des gants lorsque vous sortez ?",
#         "Décrivez ce que signifie pour vous un bon état de santé",
        "Quand avez-vous passé votre dernier examen de santé ?",
        "Avez-vous passé un test pour le coronavirus au cours des derniers mois ?",
        "Savez-vous que, pendant la durée du confinement, la meilleure façon de recevoir une aide médicale est d'appeler votre médecin, de travailler avec lui via les médias sociaux ou par appel vidéo ?",
        "Considérez-vous que vous êtes en bonne santé en ce moment ?",
    ]
)

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
feature_names = [f.name for f in features.values()]
lifestyle_df = lifestyle_df.rename(
    columns={old: new for old, new in zip(features.keys(), feature_names)}
)

# %%
dfs = []
from IPython.display import display

for feature in features.values():
    feature_df = feature.encode(lifestyle_df)
    dfs.append(feature_df)

processed_data = pd.concat(dfs, axis=1)
processed_data

# %% [markdown]
# Sanity check:

# %%
for col in processed_data.columns:
    print(col, processed_data[col].unique())

# %%
use_locations = False

if use_locations:
    locations = []
    for i, row in tqdm.tqdm(list(lifestyle_df[["city", "country"]].iterrows())):
        location = str(row["city"]) + ", " + str(row["country"])
        geocoder = geopy.geocoders.Nominatim(user_agent="pdm", timeout=10)
        query_result = geocoder.geocode(location)
        address = query_result.address if query_result is not None else None
        locations.append(address)

# %% [markdown]
# ## Some initial cluster analysis

# %%
import umap
from matplotlib import pyplot as plt

# %%
clean_data = processed_data.dropna()
data = clean_data.drop(columns=[
#     "age",
#     "gender-autre",
#     "gender-f",
#     "gender-m",
])

reducer = umap.UMAP(random_state=42, n_neighbors=3)
embedding = reducer.fit_transform(data)

fig, ax = plt.subplots(figsize=(12, 10))
sns.scatterplot(
    x=embedding[:, 0], y=embedding[:, 1], hue=clean_data.age,
    ax=ax
)
plt.setp(ax, xticks=[], yticks=[])
plt.show()
