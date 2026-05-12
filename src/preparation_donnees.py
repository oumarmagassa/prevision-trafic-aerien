
import pandas as pd

# Chargement des données
df = pd.read_csv("data/trafic_aerien.csv")

# Affichage des premières lignes
print(df.head())

# Informations générales
print(df.info())

# Valeurs manquantes
print(df.isnull().sum())
