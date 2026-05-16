import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import os

def creer_features(df):
    """Crée les variables explicatives pour XGBoost."""
    df = df.copy()
    # Lags (valeurs passées)
    df['lag_1']  = df['passagers'].shift(1)
    df['lag_3']  = df['passagers'].shift(3)
    df['lag_12'] = df['passagers'].shift(12)   # même mois année précédente
    # Moyenne mobile
    df['mm_3']   = df['passagers'].rolling(3).mean().shift(1)
    df['mm_6']   = df['passagers'].rolling(6).mean().shift(1)
    return df

def entrainer_xgboost(df):
    df = creer_features(df)
    df = df.dropna()

    features = ['mois', 'annee', 'rang_temps', 'trimestre', 'covid',
                'lag_1', 'lag_3', 'lag_12', 'mm_3', 'mm_6']
    X = df[features]
    y = df['passagers']

    # Découpage temporel : train jusqu'à fin 2022, test 2023-2024
    train = df[df['annee'] <= 2022]
    test  = df[df['annee'] > 2022]

    X_train, y_train = train[features], train['passagers']
    X_test,  y_test  = test[features],  test['passagers']

    model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae  = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = np.mean(np.abs((y_test.values - predictions) / y_test.values)) * 100

    print("\n📊 Modèle XGBoost")
    print(f"   MAE  : {mae:,.0f} passagers")
    print(f"   RMSE : {rmse:,.0f} passagers")
    print(f"   MAPE : {mape:.1f}%")
    print(f"   Période test : {test['date'].min().date()} → {test['date'].max().date()}")

    # Graphique résultats
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train['date'], y_train, label='Entraînement', color='steelblue')
    ax.plot(test['date'], y_test, label='Réel (test)', color='green')
    ax.plot(test['date'], predictions, label='Prévision XGBoost', color='red', linestyle='--')
    ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-06-01'),
               alpha=0.15, color='gray', label='Période Covid')
    ax.set_title("Prévision du trafic Air France — XGBoost")
    ax.set_xlabel("Date")
    ax.set_ylabel("Passagers")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/xgboost_predictions.png", dpi=150)
    print("   📈 Graphique sauvegardé : figures/xgboost_predictions.png")

    return model, {"MAE": mae, "RMSE": rmse, "MAPE": mape}, test, predictions

if __name__ == "__main__":
    df = pd.read_csv("data/trafic_airfrance.csv", parse_dates=['date'])
    model, metriques, test, preds = entrainer_xgboost(df)
    print("\n✅ Modèle XGBoost terminé.")
