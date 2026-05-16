import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluer_modele(y_true, y_pred, nom="Modèle"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n📊 {nom}")
    print(f"   MAE  : {mae:,.0f} passagers")
    print(f"   RMSE : {rmse:,.0f} passagers")
    print(f"   MAPE : {mape:.1f}%")
    return {"nom": nom, "MAE": mae, "RMSE": rmse, "MAPE": mape}

def modele_naif(df):
    """Prévision = valeur du même mois l'année précédente (saisonnalité 12 mois)."""
    df = df.copy()
    df['prediction_naive'] = df['passagers'].shift(12)
    df_eval = df.dropna(subset=['prediction_naive'])
    # On exclut la période Covid pour une évaluation plus juste
    df_eval = df_eval[df_eval['covid'] == 0]
    return evaluer_modele(df_eval['passagers'], df_eval['prediction_naive'], "Modèle Naïf (même mois -1 an)")

def modele_moyenne_mobile(df):
    """Prévision = moyenne des 3 derniers mois identiques."""
    df = df.copy()
    df['prediction_mm'] = df['passagers'].rolling(window=3).mean().shift(1)
    df_eval = df.dropna(subset=['prediction_mm'])
    df_eval = df_eval[df_eval['covid'] == 0]
    return evaluer_modele(df_eval['passagers'], df_eval['prediction_mm'], "Moyenne Mobile (3 mois)")

if __name__ == "__main__":
    df = pd.read_csv("data/trafic_airfrance.csv", parse_dates=['date'])
    print("=== MODÈLES DE RÉFÉRENCE ===")
    resultats = []
    resultats.append(modele_naif(df))
    resultats.append(modele_moyenne_mobile(df))
    print("\n✅ Évaluation terminée.")
