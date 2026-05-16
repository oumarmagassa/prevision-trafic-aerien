import pandas as pd
import glob
import os

def charger_donnees_brutes(dossier_data="data/raw"):
    """Charge et concatène tous les fichiers CSV DGAC."""
    files = sorted(glob.glob(os.path.join(dossier_data, "ASP_CIE_*.csv")))
    if not files:
        raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {dossier_data}")
    dfs = [pd.read_csv(f, sep=';', encoding='utf-8-sig') for f in files]
    return pd.concat(dfs, ignore_index=True)

def preparer_donnees_airfrance(df):
    """Filtre et prépare les données Air France."""
    af = df[df['CIE_NOM'] == 'AIR FRANCE'].copy()
    af['date'] = pd.to_datetime(af['ANMOIS'].astype(str), format='%Y%m')
    af = af[['date', 'CIE_PAX', 'CIE_VOL', 'CIE_PKT', 'CIE_PEQ']].rename(columns={
        'CIE_PAX': 'passagers',
        'CIE_VOL': 'vols',
        'CIE_PKT': 'passagers_km',
        'CIE_PEQ': 'passagers_equivalents'
    })
    af = af.sort_values('date').reset_index(drop=True)

    # Features temporelles utiles pour les modèles
    af['mois'] = af['date'].dt.month
    af['annee'] = af['date'].dt.year
    af['rang_temps'] = range(len(af))
    af['trimestre'] = af['date'].dt.quarter

    # Flag Covid (mars 2020 - juin 2021)
    af['covid'] = ((af['date'] >= '2020-03-01') & (af['date'] <= '2021-06-01')).astype(int)

    return af

if __name__ == "__main__":
    df_brut = charger_donnees_brutes()
    df = preparer_donnees_airfrance(df_brut)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/trafic_airfrance.csv", index=False)
    print(f"✅ Données sauvegardées : {len(df)} mois ({df['date'].min().year} - {df['date'].max().year})")
    print(df[['date', 'passagers', 'vols']].tail())
