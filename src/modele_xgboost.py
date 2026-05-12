import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

df = pd.read_csv("data/trafic_aerien.csv")
df["date"] = pd.to_datetime(df["date"])

# Création de variables temporelles
df["mois"] = df["date"].dt.month
df["annee"] = df["date"].dt.year
df["rang_temps"] = np.arange(len(df))

# Variables explicatives et cible
X = df[["mois", "annee", "rang_temps"]]
y = df["passagers"]

# Découpage temporel simple : entraînement puis test
train_size = int(len(df) * 0.8)

X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

model = XGBRegressor(
    n_estimators=50,
    max_depth=2,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"MAE XGBoost : {mae:.2f}")
print(f"RMSE XGBoost : {rmse:.2f}")
