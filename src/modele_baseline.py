
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

df = pd.read_csv("data/trafic_aerien.csv")
df["date"] = pd.to_datetime(df["date"])

# Modèle naïf : la prévision du mois suivant = valeur du mois précédent
df["prediction_naive"] = df["passagers"].shift(1)

# Suppression de la première ligne sans prédiction
df_eval = df.dropna()

mae = mean_absolute_error(df_eval["passagers"], df_eval["prediction_naive"])
rmse = np.sqrt(mean_squared_error(df_eval["passagers"], df_eval["prediction_naive"]))

print(f"MAE du modèle naïf : {mae:.2f}")
print(f"RMSE du modèle naïf : {rmse:.2f}")
