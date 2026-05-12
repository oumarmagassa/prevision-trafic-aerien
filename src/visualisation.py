import pandas as pd
import matplotlib.pyplot as plt

# Chargement des données
df = pd.read_csv("data/trafic_aerien.csv")

# Conversion de la date
df["date"] = pd.to_datetime(df["date"])

# Création du graphique
plt.figure(figsize=(10, 5))
plt.plot(df["date"], df["passagers"], marker="o")
plt.title("Évolution du trafic passagers")
plt.xlabel("Date")
plt.ylabel("Nombre de passagers")
plt.grid(True)

# Sauvegarde du graphique
plt.savefig("figures/evolution_trafic.png", bbox_inches="tight")

print("Graphique sauvegardé dans figures/evolution_trafic.png")
