import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# ─── Config page ────────────────────────────────────────────────
st.set_page_config(
    page_title="Trafic Air France",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Prévision du trafic passagers — Air France")
st.caption("Données officielles DGAC · 2010–2024 · Modèles : Naïf · XGBoost")

# ─── Chargement des données ─────────────────────────────────────
@st.cache_data
def charger_donnees():
    df = pd.read_csv("data/trafic_airfrance.csv", parse_dates=['date'])
    return df

df = charger_donnees()

# ─── Sidebar filtres ────────────────────────────────────────────
st.sidebar.header("⚙️ Paramètres")
annee_min, annee_max = int(df['date'].dt.year.min()), int(df['date'].dt.year.max())
periode = st.sidebar.slider("Période d'analyse", annee_min, annee_max, (2015, annee_max))
afficher_covid = st.sidebar.checkbox("Afficher la zone Covid", value=True)

df_filtre = df[(df['date'].dt.year >= periode[0]) & (df['date'].dt.year <= periode[1])]

# ─── KPIs ───────────────────────────────────────────────────────
st.subheader("📊 Indicateurs clés")
col1, col2, col3, col4 = st.columns(4)

total_pax    = df_filtre['passagers'].sum()
moy_mensuelle = df_filtre['passagers'].mean()
max_mois     = df_filtre.loc[df_filtre['passagers'].idxmax()]
min_mois     = df_filtre.loc[df_filtre['passagers'].idxmin()]

col1.metric("Total passagers", f"{total_pax/1e6:.1f}M")
col2.metric("Moyenne mensuelle", f"{moy_mensuelle/1e3:.0f}K")
col3.metric("🔺 Pic", f"{max_mois['passagers']/1e6:.2f}M", max_mois['date'].strftime("%b %Y"))
col4.metric("🔻 Creux", f"{min_mois['passagers']/1e3:.0f}K", min_mois['date'].strftime("%b %Y"))

st.divider()

# ─── Évolution du trafic ────────────────────────────────────────
st.subheader("📈 Évolution du trafic passagers")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df_filtre['date'], df_filtre['passagers'] / 1e6, color='steelblue', linewidth=1.8)
ax.fill_between(df_filtre['date'], df_filtre['passagers'] / 1e6, alpha=0.1, color='steelblue')
if afficher_covid:
    ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-06-01'),
               alpha=0.15, color='red', label='Période Covid')
    ax.legend()
ax.set_ylabel("Passagers (millions)")
ax.set_xlabel("")
ax.grid(True, alpha=0.3)
ax.set_title(f"Trafic mensuel Air France ({periode[0]}–{periode[1]})")
st.pyplot(fig)

# ─── Saisonnalité ───────────────────────────────────────────────
st.subheader("📅 Saisonnalité mensuelle")
df_hors_covid = df[df['covid'] == 0]
saisonnalite  = df_hors_covid.groupby('mois')['passagers'].mean() / 1e6

fig2, ax2 = plt.subplots(figsize=(10, 3))
mois_labels = ['Jan','Fév','Mar','Avr','Mai','Jun','Jul','Aoû','Sep','Oct','Nov','Déc']
bars = ax2.bar(mois_labels, saisonnalite.values, color='steelblue', alpha=0.8)
ax2.set_ylabel("Passagers moyens (millions)")
ax2.set_title("Saisonnalité moyenne (hors période Covid)")
ax2.grid(True, axis='y', alpha=0.3)
# Mettre en évidence les mois peak
peak_idx = saisonnalite.values.argmax()
bars[peak_idx].set_color('tomato')
st.pyplot(fig2)

st.divider()

# ─── Modèle XGBoost ─────────────────────────────────────────────
st.subheader("🤖 Modèle de prévision XGBoost")

@st.cache_data
def entrainer_modele(df):
    df = df.copy()
    df['lag_1']  = df['passagers'].shift(1)
    df['lag_3']  = df['passagers'].shift(3)
    df['lag_12'] = df['passagers'].shift(12)
    df['mm_3']   = df['passagers'].rolling(3).mean().shift(1)
    df['mm_6']   = df['passagers'].rolling(6).mean().shift(1)
    df = df.dropna()

    features = ['mois', 'annee', 'rang_temps', 'trimestre', 'covid',
                'lag_1', 'lag_3', 'lag_12', 'mm_3', 'mm_6']

    train = df[df['annee'] <= 2022]
    test  = df[df['annee'] > 2022]

    model = XGBRegressor(n_estimators=200, max_depth=3,
                         learning_rate=0.05, subsample=0.8, random_state=42)
    model.fit(train[features], train['passagers'])

    preds = model.predict(test[features])
    mae   = mean_absolute_error(test['passagers'], preds)
    rmse  = np.sqrt(mean_squared_error(test['passagers'], preds))
    mape  = np.mean(np.abs((test['passagers'].values - preds) / test['passagers'].values)) * 100

    return model, test, preds, mae, rmse, mape, train

model, test, preds, mae, rmse, mape, train = entrainer_modele(df)

# Métriques modèle
m1, m2, m3 = st.columns(3)
m1.metric("MAE",  f"{mae/1e3:.0f}K passagers")
m2.metric("RMSE", f"{rmse/1e3:.0f}K passagers")
m3.metric("MAPE", f"{mape:.1f}%")

# Graphique prévisions
fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.plot(train['date'], train['passagers'] / 1e6, label='Entraînement (2010–2022)', color='steelblue')
ax3.plot(test['date'],  test['passagers']  / 1e6, label='Réel (2023–2024)', color='green')
ax3.plot(test['date'],  preds              / 1e6, label='Prévision XGBoost', color='red', linestyle='--', linewidth=2)
if afficher_covid:
    ax3.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-06-01'),
                alpha=0.15, color='gray', label='Covid')
ax3.set_ylabel("Passagers (millions)")
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_title("Prévision XGBoost vs Réel")
st.pyplot(fig3)

st.divider()

# ─── Données brutes ─────────────────────────────────────────────
with st.expander("🗂️ Voir les données brutes"):
    st.dataframe(
        df_filtre[['date','passagers','vols','passagers_km']].rename(columns={
            'date': 'Date',
            'passagers': 'Passagers',
            'vols': 'Vols',
            'passagers_km': 'Passagers-km (milliards)'
        }),
        use_container_width=True
    )

st.caption("Source : DGAC — Direction Générale de l'Aviation Civile | Licence Ouverte 2.0")
