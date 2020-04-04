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

import tqdm
import attr
import pandas as pd
import numpy as np
import geopy

# %% [markdown]
# # Load data

# %%
df = pd.read_excel('data/FR-Questionnaire_francais.xlsx')

# %%
df.head(3)

# %% [markdown]
# # Refine dataset

# %%
lifestyle_df = df.drop(
    columns = [
        "Avez-vous l'impression que les gens autour de vous respectent les mesures politiques prises ?",
        "Quels sont les sentiments que vous avez éprouvés depuis le début du confinement ?",
        "Que savez-vous sur le coronavirus ?", 'Horodateur',
        "Avez-vous accès aux ressources suivantes ?",
        "Utilisez-vous un masque et/ou des gants lorsque vous sortez ?",
        "Décrivez ce que signifie pour vous un bon état de santé",
        "Quand avez-vous passé votre dernier examen de santé ?",
        "Depuis le début du confinement vous mangez en quantité :  inférieure, identique, supérieure à d'habitude ?",
        "Avez-vous passé un test pour le coronavirus au cours des derniers mois ?",
        "Savez-vous que, pendant la durée du confinement, la meilleure façon de recevoir une aide médicale est d'appeler votre médecin, de travailler avec lui via les médias sociaux ou par appel vidéo ?",
        'Considérez-vous que vous êtes en bonne santé en ce moment ?'
    ]
)

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
    def encode(self, df):
        drop_first = len(df[self.name].dropna().unique()) <= 2
        dummies = pd.get_dummies(df[self.name], drop_first=drop_first, prefix=self.name, prefix_sep="-")
        return dummies.rename(columns=process_col)


@attr.s
class OrdinalFeature(Feature):
    levels = attr.ib(default=None)
    
    def encode(self, df):
        renaming_dict = {name: ord for name, ord in zip(self.levels, range(len(self.levels)))}
        return df[self.name].replace(renaming_dict)
    
    
@attr.s
class MultiCatFeature(Feature):
    sep = attr.ib(default=", ")
    
    def encode(self, df):
        dummies = df[self.name].str.get_dummies(sep=self.sep)
        return dummies.rename(columns=process_col).rename(columns=lambda x: f"{self.name}-{x}")


@attr.s
class NumFeature(Feature):

    def encode(self, df):
        def to_num(x):
            num_str = re.sub("[^0-9]", "", str(x))
            if len(num_str) > 0:
                return float(int(num_str))
            
        return df[self.name].apply(to_num)


@attr.s
class CityFeature(Feature):
    
    def encode(self, df):
        pass


@attr.s
class CountryFeature(Feature):
    
    def encode(self, df):
        pass


# %%
features = [
    MultiCatFeature(name='information_channel'),
    OrdinalFeature(name='impact_on_life',
                   levels=["Pas d'impact", 'Léger', 'Modéré', 'Significatif']),
    MultiCatFeature(name='places_visited'),
    OrdinalFeature(name='number_exits',
                   levels=["Aucune", "Entre 1 et 3 fois", "4 à 7 fois", "Plus de 7 fois"]),
    CategoricalFeature(name='transport'),
    NumFeature(name='num_close_contact'),
    CategoricalFeature(name='recent_travel'),
    CategoricalFeature(name='volunteer'),
    MultiCatFeature(name='shared_installation'),
    CategoricalFeature(name='work_mode'),
    OrdinalFeature(name='distance_to_food_shop',
                   levels=['Moins de 15 minutes', '15-30 minutes', 'Plus de 30 minutes']),
    MultiCatFeature(name='food_shop_type'),
    OrdinalFeature(name='shopping_frequency',
                   levels=['Chaque jour', 'Tous les 3 jours environ', 'Chaque semaine', 'Toutes les deux semaines', 'Moins souvent']),
    CategoricalFeature(name='contact_frequency'),
    MultiCatFeature(name='relatives_contact_mean'),
    MultiCatFeature(name='work_contact_mean'),
    OrdinalFeature(name='relatives_contact_frequency',
                   levels=['Plusieurs fois par jour', 'Tous les jours', 'Tous les 3 jours environ', 'Toutes les semaines', 'Toutes les deux semaines', 'Pas du tout']),
    OrdinalFeature(name='num_relatives_contact',
                   levels=['Aucun', '1-3', '4-6', '7-10', '11-20', 'Plus de 20']),
    NumFeature(name='age'),
    CategoricalFeature(name='gender'),
    CityFeature(name='city'),
    CountryFeature(name='country'),
    CategoricalFeature(name='profession'),
    OrdinalFeature(name='work_exposure',
                   levels=['Faible', 'Moyen', 'Élevé']),
    NumFeature(name='household_size'),
    CategoricalFeature(name='shared_room'),
]

# %%
feature_names = [f.name for f in features]
lifestyle_df = lifestyle_df.rename(columns={
    old: new for old, new in zip(lifestyle_df.columns, feature_names)
})

# %%
dfs = []
for feature in features:
    feature_df = feature.encode(lifestyle_df)
    dfs.append(feature_df)
    
processed_data = pd.concat(dfs, axis=1)
processed_data

# %%
locations = []

for i, row in tqdm.tqdm_notebook(list(lifestyle_df[['city', 'country']].iterrows())):
    location = str(row['city']) + ', ' + str(row['country'])
    geocoder = geopy.geocoders.Nominatim(user_agent='pdm', timeout=10)
    query_result = geocoder.geocode(location)
    address = query_result.address if query_result is not None else None
    locations.append(address)
