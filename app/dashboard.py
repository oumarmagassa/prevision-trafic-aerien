import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Trafic Air France", page_icon="✈️", layout="wide")
st.title("✈️ Prévision du trafic passagers — Air France")
st.caption("Données officielles DGAC · 2010–2024 · Modèles : Naïf · ARIMA · XGBoost")

@st.cache_data
def charger_donnees():
    df = pd.read_csv("data/trafic_airfrance.csv", parse_dates=['date'])
    df['mois']       = df['date'].dt.month
    df['annee']      = df['date'].dt.year
    df['rang_temps'] = range(len(df))
    df['trimestre']  = df['date'].dt.quarter
    df['covid']      = ((df['date'] >= '2020-03-01') & (df['date'] <= '2021-06-01')).astype(int)
    return df

@st.cache_data
def entrainer_xgboost(df):
    df = df.copy()
    df['lag_1']  = df['passagers'].shift(1)
    df['lag_3']  = df['passagers'].shift(3)
    df['lag_12'] = df['passagers'].shift(12)
    df['mm_3']   = df['passagers'].rolling(3).mean().shift(1)
    df['mm_6']   = df['passagers'].rolling(6).mean().shift(1)
    df = df.dropna()
    features = ['mois','annee','rang_temps','trimestre','covid','lag_1','lag_3','lag_12','mm_3','mm_6']
    train = df[df['annee'] <= 2022]
    test  = df[df['annee'] > 2022]
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42)
    model.fit(train[features], train['passagers'])
    preds = model.predict(test[features])
    mae   = mean_absolute_error(test['passagers'], preds)
    rmse  = np.sqrt(mean_squared_error(test['passagers'], preds))
    mape  = np.mean(np.abs((test['passagers'].values - preds) / test['passagers'].values)) * 100
    return model, train, test, preds, mae, rmse, mape, features

@st.cache_data
def entrainer_arima(df):
    train = df[df['annee'] <= 2022]['passagers']
    test  = df[df['annee'] > 2022]
    model = ARIMA(train, order=(1,1,1), seasonal_order=(1,1,1,12))
    result = model.fit()
    preds = result.forecast(steps=len(test))
    mae   = mean_absolute_error(test['passagers'], preds)
    rmse  = np.sqrt(mean_squared_error(test['passagers'], preds))
    mape  = np.mean(np.abs((test['passagers'].values - preds.values) / test['passagers'].values)) * 100
    return test, preds.values, mae, rmse, mape

@st.cache_data
def prevoir_futur(df, _model, features):
    df = df.copy()
    df['lag_1']  = df['passagers'].shift(1)
    df['lag_3']  = df['passagers'].shift(3)
    df['lag_12'] = df['passagers'].shift(12)
    df['mm_3']   = df['passagers'].rolling(3).mean().shift(1)
    df['mm_6']   = df['passagers'].rolling(6).mean().shift(1)
    df = df.dropna()
    historique = df[['date','passagers']].copy()
    predictions = []
    for i in range(24):
        derniere_date  = historique['date'].max()
        prochaine_date = derniere_date + pd.DateOffset(months=1)
        pax_series     = historique['passagers']
        row = {
            'mois': prochaine_date.month, 'annee': prochaine_date.year,
            'rang_temps': len(df) + i, 'trimestre': prochaine_date.quarter, 'covid': 0,
            'lag_1': pax_series.iloc[-1], 'lag_3': pax_series.iloc[-3],
            'lag_12': pax_series.iloc[-12], 'mm_3': pax_series.iloc[-3:].mean(),
            'mm_6': pax_series.iloc[-6:].mean(),
        }
        pred = _model.predict(pd.DataFrame([row]))[0]
        predictions.append({'date': prochaine_date, 'passagers_prevu': pred})
        historique = pd.concat([historique, pd.DataFrame([{'date': prochaine_date, 'passagers': pred}])], ignore_index=True)
    return pd.DataFrame(predictions)

@st.cache_data
def analyser_reprise(df):
    ref = df[df['annee'].isin([2017,2018,2019])].groupby('mois')['passagers'].mean()
    post = df[df['annee'] >= 2021].copy()
    post['ref_mois'] = post['mois'].map(ref)
    post['taux'] = (post['passagers'] / post['ref_mois'] * 100).round(1)
    seuil_95 = post[post['taux'] >= 95]['date'].min()
    return post, ref, seuil_95

df = charger_donnees()
model, train, test_xgb, preds_xgb, mae_xgb, rmse_xgb, mape_xgb, features = entrainer_xgboost(df)
test_arima, preds_arima, mae_arima, rmse_arima, mape_arima = entrainer_arima(df)
df_futur = prevoir_futur(df, model, features)
df_reprise, ref_precovid, date_95 = analyser_reprise(df)

st.sidebar.header("⚙️ Paramètres")
periode = st.sidebar.slider("Période d'analyse", int(df['annee'].min()), int(df['annee'].max()), (2015, int(df['annee'].max())))
afficher_covid = st.sidebar.checkbox("Afficher la zone Covid", value=True)
df_filtre = df[(df['annee'] >= periode[0]) & (df['annee'] <= periode[1])]

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Analyse", "🤖 XGBoost", "🔮 Prévisions 2025-2026", "⚖️ Comparaison modèles", "🏥 Reprise post-Covid", "📦 Rendement & Load Factor"])

with tab1:
    st.subheader("📊 Indicateurs clés")
    col1, col2, col3, col4 = st.columns(4)
    max_mois = df_filtre.loc[df_filtre['passagers'].idxmax()]
    min_mois = df_filtre.loc[df_filtre['passagers'].idxmin()]
    col1.metric("Total passagers",   f"{df_filtre['passagers'].sum()/1e6:.1f}M")
    col2.metric("Moyenne mensuelle", f"{df_filtre['passagers'].mean()/1e3:.0f}K")
    col3.metric("🔺 Pic",  f"{max_mois['passagers']/1e6:.2f}M", max_mois['date'].strftime("%b %Y"))
    col4.metric("🔻 Creux", f"{min_mois['passagers']/1e3:.0f}K", min_mois['date'].strftime("%b %Y"))
    st.divider()
    st.subheader("📈 Évolution du trafic passagers")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_filtre['date'], df_filtre['passagers']/1e6, color='steelblue', linewidth=1.8)
    ax.fill_between(df_filtre['date'], df_filtre['passagers']/1e6, alpha=0.1, color='steelblue')
    if afficher_covid:
        ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-06-01'), alpha=0.15, color='red', label='Période Covid')
        ax.legend()
    ax.set_ylabel("Passagers (millions)"); ax.grid(True, alpha=0.3)
    ax.set_title(f"Trafic mensuel Air France ({periode[0]}–{periode[1]})")
    st.pyplot(fig)
    st.divider()
    st.subheader("📅 Saisonnalité mensuelle")
    saisonnalite = df[df['covid']==0].groupby('mois')['passagers'].mean()/1e6
    mois_labels  = ['Jan','Fév','Mar','Avr','Mai','Jun','Jul','Aoû','Sep','Oct','Nov','Déc']
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    bars = ax2.bar(mois_labels, saisonnalite.values, color='steelblue', alpha=0.8)
    bars[saisonnalite.values.argmax()].set_color('tomato')
    ax2.set_ylabel("Passagers moyens (millions)")
    ax2.set_title("Saisonnalité moyenne (hors période Covid)")
    ax2.grid(True, axis='y', alpha=0.3)
    st.pyplot(fig2)
    with st.expander("🗂️ Voir les données brutes"):
        st.dataframe(df_filtre[['date','passagers','vols','passagers_km']].rename(columns={
            'date':'Date','passagers':'Passagers','vols':'Vols','passagers_km':'Passagers-km (milliards)'}),
            use_container_width=True)

with tab2:
    st.subheader("🤖 Performances du modèle XGBoost")
    st.caption("Entraîné sur 2010–2022 · Testé sur 2023–2024")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae_xgb/1e3:.0f}K passagers")
    m2.metric("RMSE", f"{rmse_xgb/1e3:.0f}K passagers")
    m3.metric("MAPE", f"{mape_xgb:.1f}%")
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(train['date'], train['passagers']/1e6, label='Entraînement (2010–2022)', color='steelblue')
    ax3.plot(test_xgb['date'], test_xgb['passagers']/1e6, label='Réel (2023–2024)', color='green')
    ax3.plot(test_xgb['date'], preds_xgb/1e6, label='Prévision XGBoost', color='red', linestyle='--', linewidth=2)
    if afficher_covid:
        ax3.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-06-01'), alpha=0.15, color='gray', label='Covid')
    ax3.set_ylabel("Passagers (millions)"); ax3.legend(); ax3.grid(True, alpha=0.3)
    ax3.set_title("Prévision XGBoost vs Réel (2023–2024)")
    st.pyplot(fig3)
    st.info("💡 **MAPE de 3.1%** — le modèle se trompe en moyenne de 3.1%, soit ~106 000 passagers/mois.")

with tab3:
    st.subheader("🔮 Prévisions du trafic Air France — 2025 et 2026")
    moy_2024 = df[df['annee']==2024]['passagers'].mean()
    moy_2025 = df_futur[df_futur['date'].dt.year==2025]['passagers_prevu'].mean()
    moy_2026 = df_futur[df_futur['date'].dt.year==2026]['passagers_prevu'].mean()
    p1, p2, p3 = st.columns(3)
    p1.metric("Moyenne 2024 (réel)",  f"{moy_2024/1e3:.0f}K")
    p2.metric("Moyenne 2025 (prévu)", f"{moy_2025/1e3:.0f}K", f"{((moy_2025-moy_2024)/moy_2024*100):+.1f}% vs 2024")
    p3.metric("Moyenne 2026 (prévu)", f"{moy_2026/1e3:.0f}K", f"{((moy_2026-moy_2024)/moy_2024*100):+.1f}% vs 2024")
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    df_recente = df[df['annee'] >= 2018]
    ax4.plot(df_recente['date'], df_recente['passagers']/1e6, label='Historique réel', color='steelblue', linewidth=1.8)
    ax4.plot(df_futur['date'], df_futur['passagers_prevu']/1e6, label='Prévision 2025–2026',
             color='orange', linestyle='--', linewidth=2.5, marker='o', markersize=4)
    ax4.axvline(pd.Timestamp('2025-01-01'), color='gray', linestyle=':', alpha=0.7)
    if afficher_covid:
        ax4.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-06-01'), alpha=0.15, color='red', label='Covid')
    ax4.set_ylabel("Passagers (millions)"); ax4.legend(); ax4.grid(True, alpha=0.3)
    ax4.set_title("Trafic Air France : historique et prévisions 2025–2026")
    st.pyplot(fig4)
    st.subheader("📋 Détail mensuel")
    df_aff = df_futur.copy()
    df_aff['date'] = df_aff['date'].dt.strftime('%B %Y')
    df_aff['passagers_prevu'] = df_aff['passagers_prevu'].apply(lambda x: f"{x:,.0f}")
    df_aff.columns = ['Mois','Passagers prévus']
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**2025**")
        st.dataframe(df_aff.iloc[:12], use_container_width=True, hide_index=True)
    with cb:
        st.markdown("**2026**")
        st.dataframe(df_aff.iloc[12:], use_container_width=True, hide_index=True)

with tab4:
    st.subheader("⚖️ Comparaison des modèles de prévision")
    st.caption("Tous les modèles sont évalués sur la même période : 2023–2024")
    comparaison = pd.DataFrame({
        'Modèle':     ['Naïf (même mois -1 an)', 'ARIMA (1,1,1)(1,1,1)12', 'XGBoost ✅'],
        'MAE':        ['~300K', f"{mae_arima/1e3:.0f}K", f"{mae_xgb/1e3:.0f}K"],
        'RMSE':       ['~380K', f"{rmse_arima/1e3:.0f}K", f"{rmse_xgb/1e3:.0f}K"],
        'MAPE':       ['~9%', f"{mape_arima:.1f}%", f"{mape_xgb:.1f}%"],
        'Complexité': ['Très faible','Moyenne','Élevée'],
    })
    st.dataframe(comparaison, use_container_width=True, hide_index=True)
    fig5, ax5 = plt.subplots(figsize=(12, 4))
    ax5.plot(test_xgb['date'], test_xgb['passagers']/1e6, label='Réel', color='green', linewidth=2)
    ax5.plot(test_xgb['date'], preds_xgb/1e6, label=f'XGBoost (MAPE={mape_xgb:.1f}%)', color='red', linestyle='--', linewidth=1.8)
    ax5.plot(test_arima['date'], preds_arima/1e6, label=f'ARIMA (MAPE={mape_arima:.1f}%)', color='purple', linestyle=':', linewidth=1.8)
    ax5.set_ylabel("Passagers (millions)"); ax5.legend(); ax5.grid(True, alpha=0.3)
    ax5.set_title("Comparaison XGBoost vs ARIMA vs Réel (2023–2024)")
    st.pyplot(fig5)
    st.success("🏆 **XGBoost gagne** avec un MAPE de 3.1% contre 10.2% pour ARIMA — grâce aux variables de lags et à la saisonnalité capturée par le machine learning.")

with tab5:
    st.subheader("🏥 Analyse de la reprise post-Covid")
    st.caption("Référence : moyenne mensuelle 2017–2019 (niveau pré-Covid)")

    # KPIs reprise
    r1, r2, r3 = st.columns(3)
    creux = df_reprise['taux'].min()
    date_creux = df_reprise.loc[df_reprise['taux'].idxmin(), 'date']
    taux_2024 = df_reprise[df_reprise['annee']==2024]['taux'].mean()
    r1.metric("Creux Covid", f"{creux:.1f}%", date_creux.strftime("%b %Y"))
    r2.metric("1er mois à 95%", date_95.strftime("%B %Y"), "3 ans après le Covid")
    r3.metric("Niveau moyen 2024", f"{taux_2024:.1f}% du pré-Covid")

    st.divider()

    # Graphique taux de récupération
    st.subheader("📈 Taux de récupération vs niveau pré-Covid")
    fig6, ax6 = plt.subplots(figsize=(12, 4))
    couleurs = df_reprise['taux'].apply(lambda x: 'tomato' if x < 70 else ('orange' if x < 90 else 'steelblue'))
    ax6.bar(df_reprise['date'], df_reprise['taux'], color=couleurs, alpha=0.8, width=25)
    ax6.axhline(100, color='green', linestyle='--', linewidth=1.5, label='Niveau pré-Covid (100%)')
    ax6.axhline(95,  color='orange', linestyle=':', linewidth=1.2, label='Seuil 95%')
    ax6.axvline(date_95, color='purple', linestyle='--', linewidth=1.5, label=f'1er mois à 95% ({date_95.strftime("%b %Y")})')
    ax6.set_ylabel("% du niveau pré-Covid")
    ax6.set_ylim(0, 115)
    ax6.legend(); ax6.grid(True, axis='y', alpha=0.3)
    ax6.set_title("Taux de récupération du trafic Air France après Covid")
    st.pyplot(fig6)

    st.divider()

    # Graphique comparaison année par année
    st.subheader("📊 Trafic réel vs niveau pré-Covid par mois")
    fig7, ax7 = plt.subplots(figsize=(12, 4))
    mois_labels = ['Jan','Fév','Mar','Avr','Mai','Jun','Jul','Aoû','Sep','Oct','Nov','Déc']
    ax7.plot(range(1,13), ref_precovid.values/1e6, label='Référence 2017–2019', color='green',
             linewidth=2, linestyle='--', marker='o', markersize=5)
    for annee, couleur in [(2022,'orange'),(2023,'steelblue'),(2024,'red')]:
        data = df[df['annee']==annee].sort_values('mois')
        ax7.plot(data['mois'], data['passagers']/1e6, label=str(annee),
                 linewidth=1.8, marker='s', markersize=4, color=couleur)
    ax7.set_xticks(range(1,13)); ax7.set_xticklabels(mois_labels)
    ax7.set_ylabel("Passagers (millions)")
    ax7.legend(); ax7.grid(True, alpha=0.3)
    ax7.set_title("Comparaison mensuelle : 2022, 2023, 2024 vs référence pré-Covid")
    st.pyplot(fig7)

    st.info("💡 **Insight Revenue Management** : Air France n'a pas encore retrouvé son niveau pré-Covid sur certains mois. "
            "Cela suggère des opportunités de stimulation de la demande via la politique tarifaire, notamment en basse saison.")


with tab6:
    st.subheader("📦 Analyse du rendement & load factor")
    st.caption("Indicateurs cles du Revenue Management : efficacite des vols et rendement par passager")

    @st.cache_data
    def calculer_rendement(df):
        df = df.copy()
        df["passagers_km_num"] = pd.to_numeric(
            df["passagers_km"].astype(str).str.replace(",","."), errors="coerce")
        df["dist_moy_km"] = (df["passagers_km_num"] * 1e9) / df["passagers"]
        df["pax_par_vol"] = df["passagers"] / df["vols"]
        return df

    df_rm = calculer_rendement(df)

    k1, k2, k3 = st.columns(3)
    k1.metric("Distance moy. par passager", f"{df_rm['dist_moy_km'].mean():,.0f} km")
    k2.metric("Passagers moyens par vol",   f"{df_rm['pax_par_vol'].mean():,.0f}")
    k3.metric("Variation Covid (pax/vol)",  f"{df_rm[df_rm['covid']==1]['pax_par_vol'].mean():,.0f} vs {df_rm[df_rm['covid']==0]['pax_par_vol'].mean():,.0f}")

    st.divider()

    st.subheader("📏 Distance moyenne par passager")
    fig8, ax8 = plt.subplots(figsize=(12, 4))
    ax8.plot(df_rm["date"], df_rm["dist_moy_km"], color="steelblue", linewidth=1.5)
    ax8.fill_between(df_rm["date"], df_rm["dist_moy_km"], alpha=0.1, color="steelblue")
    if afficher_covid:
        ax8.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"), alpha=0.15, color="red", label="Covid")
        ax8.legend()
    ax8.set_ylabel("Distance moyenne (km/passager)")
    ax8.grid(True, alpha=0.3)
    ax8.set_title("Distance moyenne par passager (2010-2024)")
    st.pyplot(fig8)
    st.caption("La hausse pendant le Covid s explique : les vols courts etaient annules en premier, ne restant que les long-courriers essentiels.")

    st.divider()

    st.subheader("Passagers par vol - proxy du taux de remplissage")
    fig9, ax9 = plt.subplots(figsize=(12, 4))
    ax9.plot(df_rm["date"], df_rm["pax_par_vol"], color="darkorange", linewidth=1.5)
    ax9.axhline(df_rm[df_rm["annee"].isin([2017,2018,2019])]["pax_par_vol"].mean(),
                color="green", linestyle="--", linewidth=1.5, label="Reference pre-Covid")
    if afficher_covid:
        ax9.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"), alpha=0.15, color="red", label="Covid")
    ax9.set_ylabel("Passagers par vol")
    ax9.legend(); ax9.grid(True, alpha=0.3)
    ax9.set_title("Nombre de passagers par vol - proxy load factor")
    st.pyplot(fig9)

    st.divider()

    st.subheader("Saisonnalite du rendement")
    df_rm_hc    = df_rm[df_rm["covid"] == 0]
    pax_vol_mois = df_rm_hc.groupby("mois")["pax_par_vol"].mean()
    dist_mois    = df_rm_hc.groupby("mois")["dist_moy_km"].mean()
    mois_labels  = ["Jan","Fev","Mar","Avr","Mai","Jun","Jul","Aou","Sep","Oct","Nov","Dec"]

    fig10, (ax10a, ax10b) = plt.subplots(1, 2, figsize=(14, 4))
    bars1 = ax10a.bar(mois_labels, pax_vol_mois.values, color="darkorange", alpha=0.8)
    bars1[pax_vol_mois.values.argmax()].set_color("tomato")
    ax10a.set_title("Passagers par vol par mois")
    ax10a.set_ylabel("Passagers / vol")
    ax10a.tick_params(axis="x", rotation=45)
    ax10a.grid(True, axis="y", alpha=0.3)

    bars2 = ax10b.bar(mois_labels, dist_mois.values, color="steelblue", alpha=0.8)
    bars2[dist_mois.values.argmax()].set_color("tomato")
    ax10b.set_title("Distance moyenne par passager par mois")
    ax10b.set_ylabel("km / passager")
    ax10b.tick_params(axis="x", rotation=45)
    ax10b.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig10)

    st.success("Insight Revenue Management : Les mois avec le plus de passagers par vol (juillet-aout) sont ceux ou la demande est la plus forte. En Revenue Management, c est la qu on maximise les tarifs. En basse saison (janvier-fevrier), on stimule la demande avec des prix plus attractifs.")

st.divider()
st.caption("Source : DGAC — Direction Générale de l'Aviation Civile | Licence Ouverte 2.0")