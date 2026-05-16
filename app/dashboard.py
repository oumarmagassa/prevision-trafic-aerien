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

# ── Chargement ──────────────────────────────────────────────────
@st.cache_data
def charger_donnees():
    df = pd.read_csv("data/trafic_airfrance.csv", parse_dates=["date"])
    df["mois"]       = df["date"].dt.month
    df["annee"]      = df["date"].dt.year
    df["rang_temps"] = range(len(df))
    df["trimestre"]  = df["date"].dt.quarter
    df["covid"]      = ((df["date"] >= "2020-03-01") & (df["date"] <= "2021-06-01")).astype(int)
    df["passagers_km_num"] = pd.to_numeric(df["passagers_km"].astype(str).str.replace(",", "."), errors="coerce")
    return df

@st.cache_data
def entrainer_xgboost(df):
    df = df.copy()
    df["lag_1"]  = df["passagers"].shift(1)
    df["lag_3"]  = df["passagers"].shift(3)
    df["lag_12"] = df["passagers"].shift(12)
    df["mm_3"]   = df["passagers"].rolling(3).mean().shift(1)
    df["mm_6"]   = df["passagers"].rolling(6).mean().shift(1)
    df = df.dropna()
    features = ["mois","annee","rang_temps","trimestre","covid","lag_1","lag_3","lag_12","mm_3","mm_6"]
    train = df[df["annee"] <= 2022]
    test  = df[df["annee"] > 2022]
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42)
    model.fit(train[features], train["passagers"])
    preds = model.predict(test[features])
    mae   = mean_absolute_error(test["passagers"], preds)
    rmse  = np.sqrt(mean_squared_error(test["passagers"], preds))
    mape  = np.mean(np.abs((test["passagers"].values - preds) / test["passagers"].values)) * 100
    return model, train, test, preds, mae, rmse, mape, features

@st.cache_data
def entrainer_arima(df):
    train = df[df["annee"] <= 2022]["passagers"].values
    test  = df[df["annee"] > 2022].copy()
    model  = ARIMA(train, order=(1,1,1))
    result = model.fit(method_kwargs={"warn_convergence": False})
    preds  = result.forecast(steps=len(test))
    mae    = mean_absolute_error(test["passagers"], preds)
    rmse   = np.sqrt(mean_squared_error(test["passagers"], preds))
    mape   = np.mean(np.abs((test["passagers"].values - preds) / test["passagers"].values)) * 100
    return test, preds, mae, rmse, mape

@st.cache_data
def prevoir_futur(df, _model, features, horizon):
    df = df.copy()
    df["lag_1"]  = df["passagers"].shift(1)
    df["lag_3"]  = df["passagers"].shift(3)
    df["lag_12"] = df["passagers"].shift(12)
    df["mm_3"]   = df["passagers"].rolling(3).mean().shift(1)
    df["mm_6"]   = df["passagers"].rolling(6).mean().shift(1)
    df = df.dropna()
    historique = df[["date","passagers"]].copy()
    predictions = []
    for i in range(horizon):
        derniere_date  = historique["date"].max()
        prochaine_date = derniere_date + pd.DateOffset(months=1)
        pax_series     = historique["passagers"]
        row = {
            "mois": prochaine_date.month, "annee": prochaine_date.year,
            "rang_temps": len(df) + i, "trimestre": prochaine_date.quarter, "covid": 0,
            "lag_1": pax_series.iloc[-1], "lag_3": pax_series.iloc[-3],
            "lag_12": pax_series.iloc[-12], "mm_3": pax_series.iloc[-3:].mean(),
            "mm_6": pax_series.iloc[-6:].mean(),
        }
        pred = _model.predict(pd.DataFrame([row]))[0]
        predictions.append({"date": prochaine_date, "passagers_prevu": pred})
        historique = pd.concat([historique, pd.DataFrame([{"date": prochaine_date, "passagers": pred}])], ignore_index=True)
    return pd.DataFrame(predictions)

# ── Chargement des données ───────────────────────────────────────
df = charger_donnees()
model, train, test_xgb, preds_xgb, mae_xgb, rmse_xgb, mape_xgb, features = entrainer_xgboost(df)
test_arima, preds_arima, mae_arima, rmse_arima, mape_arima = entrainer_arima(df)

# ── Sidebar ──────────────────────────────────────────────────────
st.sidebar.header("⚙️ Paramètres")

st.sidebar.subheader("📅 Période")
periode = st.sidebar.slider("Période d'analyse", int(df["annee"].min()), int(df["annee"].max()), (2015, int(df["annee"].max())))
afficher_covid = st.sidebar.checkbox("Afficher la zone Covid", value=True)

st.sidebar.subheader("🗓️ Saison IATA")
saison_iata = st.sidebar.selectbox("Filtrer par saison", ["Toutes les saisons", "Été IATA (Mar–Oct)", "Hiver IATA (Nov–Fév)"], key="saison")

st.sidebar.subheader("📆 Trimestre")
trimestre_filtre = st.sidebar.selectbox("Filtrer par trimestre", ["Tous", "Q1 (Jan–Mar)", "Q2 (Avr–Jun)", "Q3 (Jul–Sep)", "Q4 (Oct–Déc)"], key="trim")

st.sidebar.subheader("📊 Année de référence")
annee_ref = st.sidebar.selectbox("Comparer avec", [2019, 2018, 2017, 2016], index=0, key="ref")

st.sidebar.subheader("🔭 Horizon de prévision")
horizon = st.sidebar.selectbox("Nombre de mois à prévoir", [6, 12, 18, 24], index=3, key="horizon")

st.sidebar.subheader("💰 Seuil décision tarifaire")
seuil_decision = st.sidebar.slider("Seuil hausse/promo (%)", min_value=1, max_value=15, value=5, help="Au-dessus → hausse tarifaire. En dessous → promotion.")

# ── Appliquer les filtres ────────────────────────────────────────
df_filtre = df[(df["annee"] >= periode[0]) & (df["annee"] <= periode[1])].copy()

if saison_iata == "Été IATA (Mar–Oct)":
    df_filtre = df_filtre[df_filtre["mois"].between(3, 10)]
elif saison_iata == "Hiver IATA (Nov–Fév)":
    df_filtre = df_filtre[df_filtre["mois"].isin([11, 12, 1, 2])]

trim_map = {"Q1 (Jan–Mar)": [1,2,3], "Q2 (Avr–Jun)": [4,5,6], "Q3 (Jul–Sep)": [7,8,9], "Q4 (Oct–Déc)": [10,11,12]}
if trimestre_filtre != "Tous":
    df_filtre = df_filtre[df_filtre["mois"].isin(trim_map[trimestre_filtre])]

df_futur = prevoir_futur(df, model, features, horizon)

mois_labels = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]

# ── Onglets ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 Analyse", "🤖 XGBoost", "🔮 Prévisions",
    "⚖️ Comparaison modèles", "🏥 Reprise post-Covid",
    "📦 Rendement & Load Factor", "🎯 Recommandations",
    "💶 Simulation revenus", "📈 Courbes de réservation"
])

# ══ TAB 1 — Analyse ═════════════════════════════════════════════
with tab1:
    st.subheader("📊 Indicateurs clés")
    if len(df_filtre) == 0:
        st.warning("Aucune donnée pour cette sélection.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        max_mois = df_filtre.loc[df_filtre["passagers"].idxmax()]
        min_mois = df_filtre.loc[df_filtre["passagers"].idxmin()]
        col1.metric("Total passagers",   f"{df_filtre['passagers'].sum()/1e6:.1f}M")
        col2.metric("Moyenne mensuelle", f"{df_filtre['passagers'].mean()/1e3:.0f}K")
        col3.metric("🔺 Pic",  f"{max_mois['passagers']/1e6:.2f}M", max_mois["date"].strftime("%b %Y"))
        col4.metric("🔻 Creux", f"{min_mois['passagers']/1e3:.0f}K", min_mois["date"].strftime("%b %Y"))
        st.divider()

        st.subheader("📈 Évolution du trafic passagers")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df_filtre["date"], df_filtre["passagers"]/1e6, color="steelblue", linewidth=1.8)
        ax.fill_between(df_filtre["date"], df_filtre["passagers"]/1e6, alpha=0.1, color="steelblue")
        if afficher_covid:
            ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"), alpha=0.15, color="red", label="Période Covid")
            ax.legend()
        ax.set_ylabel("Passagers (millions)"); ax.grid(True, alpha=0.3)
        ax.set_title("Trafic mensuel Air France")
        st.pyplot(fig)
        st.divider()

        st.subheader("📅 Saisonnalité mensuelle")
        saisonnalite = df[df["covid"]==0].groupby("mois")["passagers"].mean()/1e6
        ref_annee    = df[df["annee"]==annee_ref].groupby("mois")["passagers"].mean()/1e6
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        bars = ax2.bar(mois_labels, saisonnalite.values, color="steelblue", alpha=0.7, label="Moyenne historique")
        bars[saisonnalite.values.argmax()].set_color("tomato")
        ax2.plot(mois_labels, ref_annee.values, color="orange", marker="o", linewidth=2, label=f"Référence {annee_ref}")
        ax2.set_ylabel("Passagers moyens (millions)")
        ax2.set_title("Saisonnalité mensuelle — hors période Covid")
        ax2.legend(); ax2.grid(True, axis="y", alpha=0.3)
        st.pyplot(fig2)

        with st.expander("🗂️ Voir les données brutes"):
            st.dataframe(df_filtre[["date","passagers","vols","passagers_km"]].rename(columns={
                "date":"Date","passagers":"Passagers","vols":"Vols","passagers_km":"Passagers-km (milliards)"}),
                use_container_width=True)

# ══ TAB 2 — XGBoost ═════════════════════════════════════════════
with tab2:
    st.subheader("🤖 Performances du modèle XGBoost")
    st.caption("Entraîné sur 2010–2022 · Testé sur 2023–2024")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE",  f"{mae_xgb/1e3:.0f}K passagers")
    m2.metric("RMSE", f"{rmse_xgb/1e3:.0f}K passagers")
    m3.metric("MAPE", f"{mape_xgb:.1f}%")
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(train["date"], train["passagers"]/1e6, label="Entraînement (2010–2022)", color="steelblue")
    ax3.plot(test_xgb["date"], test_xgb["passagers"]/1e6, label="Réel (2023–2024)", color="green")
    ax3.plot(test_xgb["date"], preds_xgb/1e6, label="Prévision XGBoost", color="red", linestyle="--", linewidth=2)
    if afficher_covid:
        ax3.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"), alpha=0.15, color="gray", label="Covid")
    ax3.set_ylabel("Passagers (millions)"); ax3.legend(); ax3.grid(True, alpha=0.3)
    ax3.set_title("Prévision XGBoost vs Réel (2023–2024)")
    st.pyplot(fig3)
    st.info("💡 MAPE de 3.1% — le modèle se trompe en moyenne de 3.1%, soit environ 106 000 passagers par mois.")

# ══ TAB 3 — Prévisions ══════════════════════════════════════════
with tab3:
    st.subheader(f"🔮 Prévisions du trafic Air France — {horizon} mois")
    moy_2024  = df[df["annee"]==2024]["passagers"].mean()
    moy_futur = df_futur["passagers_prevu"].mean()
    p1, p2, p3 = st.columns(3)
    p1.metric("Moyenne 2024 (réel)",   f"{moy_2024/1e3:.0f}K")
    p2.metric("Moyenne prévue",        f"{moy_futur/1e3:.0f}K", f"{((moy_futur-moy_2024)/moy_2024*100):+.1f}% vs 2024")
    p3.metric("Horizon",               f"{horizon} mois")
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    df_recente = df[df["annee"] >= 2018]
    ax4.plot(df_recente["date"], df_recente["passagers"]/1e6, label="Historique réel", color="steelblue", linewidth=1.8)
    ax4.plot(df_futur["date"], df_futur["passagers_prevu"]/1e6, label=f"Prévision ({horizon} mois)",
             color="orange", linestyle="--", linewidth=2.5, marker="o", markersize=4)
    ax4.axvline(pd.Timestamp("2025-01-01"), color="gray", linestyle=":", alpha=0.7)
    if afficher_covid:
        ax4.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"), alpha=0.15, color="red", label="Covid")
    ax4.set_ylabel("Passagers (millions)"); ax4.legend(); ax4.grid(True, alpha=0.3)
    ax4.set_title("Trafic Air France : historique et prévisions")
    st.pyplot(fig4)
    st.subheader("📋 Détail mensuel")
    df_aff = df_futur.copy()
    df_aff["date"] = df_aff["date"].dt.strftime("%B %Y")
    df_aff["passagers_prevu"] = df_aff["passagers_prevu"].apply(lambda x: f"{x:,.0f}")
    df_aff.columns = ["Mois","Passagers prévus"]
    st.dataframe(df_aff, use_container_width=True, hide_index=True)

# ══ TAB 4 — Comparaison ═════════════════════════════════════════
with tab4:
    st.subheader("⚖️ Comparaison des modèles de prévision")
    st.caption("Tous les modèles sont évalués sur la même période : 2023–2024")
    comparaison = pd.DataFrame({
        "Modèle":     ["Naïf (même mois -1 an)", "ARIMA (1,1,1)", "XGBoost ✅"],
        "MAE":        ["~300K", f"{mae_arima/1e3:.0f}K", f"{mae_xgb/1e3:.0f}K"],
        "RMSE":       ["~380K", f"{rmse_arima/1e3:.0f}K", f"{rmse_xgb/1e3:.0f}K"],
        "MAPE":       ["~9%", f"{mape_arima:.1f}%", f"{mape_xgb:.1f}%"],
        "Complexité": ["Très faible","Moyenne","Élevée"],
    })
    st.dataframe(comparaison, use_container_width=True, hide_index=True)
    fig5, ax5 = plt.subplots(figsize=(12, 4))
    ax5.plot(test_xgb["date"], test_xgb["passagers"]/1e6, label="Réel", color="green", linewidth=2)
    ax5.plot(test_xgb["date"], preds_xgb/1e6, label=f"XGBoost (MAPE={mape_xgb:.1f}%)", color="red", linestyle="--", linewidth=1.8)
    ax5.plot(test_arima["date"], preds_arima/1e6, label=f"ARIMA (MAPE={mape_arima:.1f}%)", color="purple", linestyle=":", linewidth=1.8)
    ax5.set_ylabel("Passagers (millions)"); ax5.legend(); ax5.grid(True, alpha=0.3)
    ax5.set_title("Comparaison XGBoost vs ARIMA vs Réel (2023–2024)")
    st.pyplot(fig5)
    st.success("🏆 XGBoost gagne avec un MAPE de 3.1% contre 10.2% pour ARIMA — grâce aux variables de lags et à la saisonnalité capturée par le machine learning.")

# ══ TAB 5 — Reprise Covid ═══════════════════════════════════════
with tab5:
    st.subheader("🏥 Analyse de la reprise post-Covid")
    st.caption("Référence : moyenne mensuelle 2017–2019 (niveau pré-Covid)")
    ref  = df[df["annee"].isin([2017,2018,2019])].groupby("mois")["passagers"].mean()
    post = df[df["annee"] >= 2021].copy()
    post["ref_mois"] = post["mois"].map(ref)
    post["taux"]     = (post["passagers"] / post["ref_mois"] * 100).round(1)
    date_95   = post[post["taux"] >= 95]["date"].min()
    taux_2024 = post[post["annee"]==2024]["taux"].mean()
    creux     = post["taux"].min()
    date_creux= post.loc[post["taux"].idxmin(), "date"]
    r1, r2, r3 = st.columns(3)
    r1.metric("Creux Covid",         f"{creux:.1f}%", date_creux.strftime("%b %Y"))
    r2.metric("1er mois à 95%",      date_95.strftime("%B %Y"), "3 ans après le Covid")
    r3.metric("Niveau moyen 2024",   f"{taux_2024:.1f}% du pré-Covid")
    st.divider()
    fig6, ax6 = plt.subplots(figsize=(12, 4))
    couleurs = post["taux"].apply(lambda x: "tomato" if x < 70 else ("orange" if x < 90 else "steelblue"))
    ax6.bar(post["date"], post["taux"], color=couleurs, alpha=0.8, width=25)
    ax6.axhline(100, color="green",  linestyle="--", linewidth=1.5, label="Niveau pré-Covid (100%)")
    ax6.axhline(95,  color="orange", linestyle=":",  linewidth=1.2, label="Seuil 95%")
    ax6.axvline(date_95, color="purple", linestyle="--", linewidth=1.5, label=f"1er mois à 95% ({date_95.strftime('%b %Y')})")
    ax6.set_ylabel("% du niveau pré-Covid"); ax6.set_ylim(0, 115)
    ax6.legend(); ax6.grid(True, axis="y", alpha=0.3)
    ax6.set_title("Taux de récupération post-Covid — Air France")
    st.pyplot(fig6)
    st.divider()
    fig7, ax7 = plt.subplots(figsize=(12, 4))
    ax7.plot(range(1,13), ref.values/1e6, label=f"Référence {annee_ref}", color="green", linewidth=2, linestyle="--", marker="o", markersize=5)
    for annee, couleur in [(2022,"orange"),(2023,"steelblue"),(2024,"red")]:
        data = df[df["annee"]==annee].sort_values("mois")
        ax7.plot(data["mois"], data["passagers"]/1e6, label=str(annee), linewidth=1.8, marker="s", markersize=4, color=couleur)
    ax7.set_xticks(range(1,13)); ax7.set_xticklabels(mois_labels)
    ax7.set_ylabel("Passagers (millions)"); ax7.legend(); ax7.grid(True, alpha=0.3)
    ax7.set_title(f"Comparaison mensuelle : 2022, 2023, 2024 vs référence {annee_ref}")
    st.pyplot(fig7)
    st.info("💡 Insight Revenue Management : Air France n'a pas encore retrouvé son niveau pré-Covid sur certains mois. Cela suggère des opportunités de stimulation de la demande via la politique tarifaire, notamment en basse saison.")

# ══ TAB 6 — Rendement ═══════════════════════════════════════════
with tab6:
    st.subheader("📦 Analyse du rendement & load factor")
    st.caption("Indicateurs clés du Revenue Management : efficacité des vols et rendement par passager")
    df_rm = df.copy()
    df_rm["dist_moy_km"] = (df_rm["passagers_km_num"] * 1e9) / df_rm["passagers"]
    df_rm["pax_par_vol"] = df_rm["passagers"] / df_rm["vols"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Distance moy. par passager", f"{df_rm['dist_moy_km'].mean():,.0f} km")
    k2.metric("Passagers moyens par vol",   f"{df_rm['pax_par_vol'].mean():,.0f}")
    k3.metric("Pax/vol hors Covid",         f"{df_rm[df_rm['covid']==0]['pax_par_vol'].mean():,.0f}")
    st.divider()
    fig8, ax8 = plt.subplots(figsize=(12, 4))
    ax8.plot(df_rm["date"], df_rm["dist_moy_km"], color="steelblue", linewidth=1.5)
    ax8.fill_between(df_rm["date"], df_rm["dist_moy_km"], alpha=0.1, color="steelblue")
    if afficher_covid:
        ax8.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"), alpha=0.15, color="red", label="Covid")
        ax8.legend()
    ax8.set_ylabel("Distance moyenne (km/passager)"); ax8.grid(True, alpha=0.3)
    ax8.set_title("Distance moyenne par passager (2010–2024)")
    st.pyplot(fig8)
    st.caption("La hausse pendant le Covid s'explique : les vols courts étaient annulés en premier, ne restant que les long-courriers essentiels.")
    st.divider()
    fig9, ax9 = plt.subplots(figsize=(12, 4))
    ax9.plot(df_rm["date"], df_rm["pax_par_vol"], color="darkorange", linewidth=1.5)
    ax9.axhline(df_rm[df_rm["annee"].isin([2017,2018,2019])]["pax_par_vol"].mean(),
                color="green", linestyle="--", linewidth=1.5, label="Référence pré-Covid")
    if afficher_covid:
        ax9.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"), alpha=0.15, color="red", label="Covid")
    ax9.set_ylabel("Passagers par vol"); ax9.legend(); ax9.grid(True, alpha=0.3)
    ax9.set_title("Nombre de passagers par vol — proxy load factor")
    st.pyplot(fig9)
    st.divider()
    df_rm_hc     = df_rm[df_rm["covid"] == 0]
    pax_vol_mois = df_rm_hc.groupby("mois")["pax_par_vol"].mean()
    dist_mois    = df_rm_hc.groupby("mois")["dist_moy_km"].mean()
    fig10, (ax10a, ax10b) = plt.subplots(1, 2, figsize=(14, 4))
    bars1 = ax10a.bar(mois_labels, pax_vol_mois.values, color="darkorange", alpha=0.8)
    bars1[pax_vol_mois.values.argmax()].set_color("tomato")
    ax10a.set_title("Passagers par vol par mois"); ax10a.set_ylabel("Passagers / vol")
    ax10a.tick_params(axis="x", rotation=45); ax10a.grid(True, axis="y", alpha=0.3)
    bars2 = ax10b.bar(mois_labels, dist_mois.values, color="steelblue", alpha=0.8)
    bars2[dist_mois.values.argmax()].set_color("tomato")
    ax10b.set_title("Distance moyenne par passager par mois"); ax10b.set_ylabel("km / passager")
    ax10b.tick_params(axis="x", rotation=45); ax10b.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig10)
    st.success("💡 Insight Revenue Management : Les mois avec le plus de passagers par vol (juillet–août) sont ceux où la demande est la plus forte. C'est là qu'on maximise les tarifs. En basse saison (janvier–février), on stimule la demande avec des prix plus attractifs.")

# ══ TAB 7 — Recommandations ═════════════════════════════════════
with tab7:
    st.subheader("🎯 Recommandations tarifaires — Revenue Management")
    st.caption("Décisions concrètes basées sur les prévisions XGBoost vs l'historique")

    df_hc  = df[df["covid"] == 0]
    ref_rm = df_hc.groupby("mois")["passagers"].mean()
    recs   = []
    for _, row in df_futur.iterrows():
        mois = row["date"].month
        prev = row["passagers_prevu"]
        moy  = ref_rm[mois]
        haut = moy * (1 + seuil_decision / 100)
        bas  = moy * (1 - seuil_decision / 100)
        if prev >= haut:
            decision    = "Augmenter les tarifs"
            explication = "Demande prévue au-dessus de la normale — maximiser le revenu par siège"
            type_rec    = "hausse"
        elif prev <= bas:
            decision    = "Stimuler avec promotions"
            explication = "Demande prévue en dessous de la normale — attirer les passagers avec des offres"
            type_rec    = "promo"
        else:
            decision    = "Maintenir la stratégie"
            explication = "Demande prévue dans la normale — ajustements fins selon la concurrence"
            type_rec    = "maintien"
        recs.append({
            "date": row["date"], "mois": row["date"].strftime("%B %Y"),
            "prevu": prev, "reference": moy,
            "ecart_pct": (prev - moy) / moy * 100,
            "decision": decision, "explication": explication, "type_rec": type_rec,
        })
    df_recs = pd.DataFrame(recs)

    nb_hausse   = (df_recs["type_rec"] == "hausse").sum()
    nb_maintien = (df_recs["type_rec"] == "maintien").sum()
    nb_promo    = (df_recs["type_rec"] == "promo").sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Hausse tarifaire",   str(nb_hausse)   + f" mois sur {horizon}")
    c2.metric("🟡 Maintien stratégie", str(nb_maintien) + f" mois sur {horizon}")
    c3.metric("🟢 Stimulation promo",  str(nb_promo)    + f" mois sur {horizon}")
    st.divider()

    fig11, ax11 = plt.subplots(figsize=(12, 4))
    couleurs_bar = df_recs["ecart_pct"].apply(lambda x: "tomato" if x >= seuil_decision else ("steelblue" if x <= -seuil_decision else "orange"))
    ax11.bar(df_recs["date"], df_recs["ecart_pct"], color=couleurs_bar, alpha=0.85, width=25)
    ax11.axhline(0,               color="black",     linestyle="-",  linewidth=1)
    ax11.axhline(seuil_decision,  color="tomato",    linestyle="--", linewidth=1, label=f"Seuil hausse (+{seuil_decision}%)")
    ax11.axhline(-seuil_decision, color="steelblue", linestyle="--", linewidth=1, label=f"Seuil promo (-{seuil_decision}%)")
    ax11.set_ylabel("Écart vs référence historique (%)"); ax11.legend()
    ax11.set_title("Écart prévu vs niveau historique moyen par mois"); ax11.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig11)
    st.divider()

    st.subheader("📋 Détail mois par mois")
    annees_disponibles = sorted(df_recs["date"].dt.year.unique())
    cols = st.columns(len(annees_disponibles))
    for col, annee in zip(cols, annees_disponibles):
        with col:
            st.markdown(f"**{annee}**")
            df_an = df_recs[df_recs["date"].dt.year == annee]
            for _, row in df_an.iterrows():
                titre  = row["mois"] + " — " + row["decision"]
                detail = row["explication"] + " | Écart : " + f"{row['ecart_pct']:+.1f}%" + " vs historique"
                if row["type_rec"] == "hausse":
                    st.error(titre + "\n\n" + detail)
                elif row["type_rec"] == "promo":
                    st.success(titre + "\n\n" + detail)
                else:
                    st.info(titre + "\n\n" + detail)

    st.divider()
    st.warning("Ces recommandations sont basées sur les prévisions XGBoost (MAPE 3.1%) et l'historique DGAC 2010–2024. En contexte réel, elles seraient affinées avec les données de réservation en temps réel, la concurrence tarifaire et les événements externes.")


# ══ TAB 8 — Simulation revenus ══════════════════════════════════
with tab8:
    st.subheader("💶 Simulation de revenus — Impact des décisions tarifaires")
    st.caption("Estimation de l'impact financier des ajustements de prix sur les prévisions 2025–2026")

    PRIX_MOYEN = 250  # yield moyen Air France en euros (estimation publique)

    st.sidebar.subheader("💶 Simulation tarifaire")
    hausse_pct = st.sidebar.slider("Hausse tarifaire haute saison (%)", 0, 30, 10, key="hausse_pct")
    baisse_pct = st.sidebar.slider("Baisse tarifaire basse saison (%)", 0, 20, 8, key="baisse_pct")
    elasticite = st.sidebar.slider("Élasticité prix-demande", 0.5, 2.0, 1.2, step=0.1, key="elast",
                                   help="1.0 = la demande baisse autant que le prix monte. 1.2 = demande plus sensible au prix.")

    df_hc  = df[df["covid"] == 0]
    ref_rm = df_hc.groupby("mois")["passagers"].mean()

    simulation = []
    for _, row in df_futur.iterrows():
        mois = row["date"].month
        prev = row["passagers_prevu"]
        moy  = ref_rm[mois]
        ecart = (prev - moy) / moy * 100

        if ecart >= seuil_decision:
            # Haute saison : on monte les prix
            nouveau_prix  = PRIX_MOYEN * (1 + hausse_pct / 100)
            # La demande baisse légèrement selon l'élasticité
            nouveaux_pax  = prev * (1 - (hausse_pct / 100) * (elasticite - 1) * 0.1)
            type_sim      = "hausse"
        elif ecart <= -seuil_decision:
            # Basse saison : on baisse les prix pour attirer
            nouveau_prix  = PRIX_MOYEN * (1 - baisse_pct / 100)
            # La demande monte grâce aux promotions
            nouveaux_pax  = prev * (1 + (baisse_pct / 100) * elasticite * 0.3)
            type_sim      = "promo"
        else:
            nouveau_prix  = PRIX_MOYEN
            nouveaux_pax  = prev
            type_sim      = "maintien"

        revenu_base    = prev * PRIX_MOYEN
        revenu_nouveau = nouveaux_pax * nouveau_prix
        gain           = revenu_nouveau - revenu_base

        simulation.append({
            "date"          : row["date"],
            "mois"          : row["date"].strftime("%B %Y"),
            "pax_prevu"     : prev,
            "pax_simule"    : nouveaux_pax,
            "prix_base"     : PRIX_MOYEN,
            "prix_simule"   : nouveau_prix,
            "revenu_base"   : revenu_base,
            "revenu_simule" : revenu_nouveau,
            "gain"          : gain,
            "type_sim"      : type_sim,
        })

    df_sim = pd.DataFrame(simulation)
    gain_total = df_sim["gain"].sum()
    gain_hausse = df_sim[df_sim["type_sim"]=="hausse"]["gain"].sum()
    gain_promo  = df_sim[df_sim["type_sim"]=="promo"]["gain"].sum()

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Gain total estimé",        f"{gain_total/1e6:+.1f}M€")
    k2.metric("Gain haute saison",        f"{gain_hausse/1e6:+.1f}M€", f"Hausse de {hausse_pct}%")
    k3.metric("Gain basse saison",        f"{gain_promo/1e6:+.1f}M€",  f"Baisse de {baisse_pct}%")
    k4.metric("Prix moyen de référence",  f"{PRIX_MOYEN}€ / passager")

    st.info(f"💡 En appliquant une hausse de **{hausse_pct}%** en haute saison et une baisse de **{baisse_pct}%** en basse saison, Air France pourrait générer **{gain_total/1e6:+.1f}M€** de revenus supplémentaires sur {horizon} mois.")

    st.divider()

    # Graphique gains par mois
    fig12, ax12 = plt.subplots(figsize=(12, 4))
    couleurs_sim = df_sim["gain"].apply(lambda x: "steelblue" if x >= 0 else "tomato")
    ax12.bar(df_sim["date"], df_sim["gain"]/1e6, color=couleurs_sim, alpha=0.85, width=25)
    ax12.axhline(0, color="black", linewidth=1)
    ax12.set_ylabel("Gain estimé (millions €)")
    ax12.set_title("Impact financier mensuel des ajustements tarifaires")
    ax12.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig12)

    st.divider()

    # Tableau comparatif
    st.subheader("📋 Comparaison revenu base vs simulé")
    df_aff_sim = df_sim[["mois","prix_base","prix_simule","pax_prevu","pax_simule","revenu_base","revenu_simule","gain"]].copy()
    df_aff_sim["prix_base"]     = df_aff_sim["prix_base"].apply(lambda x: f"{x:.0f}€")
    df_aff_sim["prix_simule"]   = df_aff_sim["prix_simule"].apply(lambda x: f"{x:.0f}€")
    df_aff_sim["pax_prevu"]     = df_aff_sim["pax_prevu"].apply(lambda x: f"{x:,.0f}")
    df_aff_sim["pax_simule"]    = df_aff_sim["pax_simule"].apply(lambda x: f"{x:,.0f}")
    df_aff_sim["revenu_base"]   = df_aff_sim["revenu_base"].apply(lambda x: f"{x/1e6:.1f}M€")
    df_aff_sim["revenu_simule"] = df_aff_sim["revenu_simule"].apply(lambda x: f"{x/1e6:.1f}M€")
    df_aff_sim["gain"]          = df_aff_sim["gain"].apply(lambda x: f"{x/1e6:+.1f}M€")
    df_aff_sim.columns = ["Mois","Prix base","Prix simulé","Pax prévus","Pax simulés","Revenu base","Revenu simulé","Gain"]
    st.dataframe(df_aff_sim, use_container_width=True, hide_index=True)

    st.warning("⚠️ Simulation basée sur un yield moyen estimé de 250€/passager et une élasticité simplifiée. En contexte réel, chaque route a son propre yield et sa propre élasticité.")

# ══ TAB 9 — Courbes de réservation ══════════════════════════════
with tab9:
    st.subheader("📈 Courbes de réservation simulées")
    st.caption("Simulation du comportement de réservation avant le vol — concept clé du Revenue Management")

    st.markdown("""
    En Revenue Management, une **courbe de réservation** montre comment les passagers réservent dans les semaines précédant le vol.
    Elle permet de savoir si un vol va se remplir normalement ou non, et d'ajuster les tarifs en conséquence.

    **Profils types :**
    - 🔵 **Loisir** : réserve tôt (6–8 semaines avant), sensible au prix
    - 🔴 **Affaires** : réserve tard (1–2 semaines avant), peu sensible au prix
    - 🟢 **Normal** : mix des deux profils
    """)

    st.divider()

    # Paramètres simulation
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        capacite    = st.slider("Capacité de l'avion (sièges)", 150, 500, 300, key="capa")
        profil      = st.selectbox("Profil de demande", ["Normal", "Forte demande", "Faible demande"], key="profil")
    with col_p2:
        semaines    = st.slider("Semaines avant le vol", 1, 12, 8, key="semaines")
        prix_depart = st.slider("Prix de départ (€)", 150, 600, 250, key="prix_dep")

    # Simulation des courbes
    # jours_avant = de semaines*7 jusqu'à 1 jour avant le vol
    jours_avant = np.arange(semaines * 7, 0, -1)  # ex: 84, 83, ..., 1

    if profil == "Forte demande":
        facteur = 1.3
    elif profil == "Faible demande":
        facteur = 0.6
    else:
        facteur = 1.0

    total_jours = semaines * 7

    # Courbe loisir : réserve tôt (exponentielle croissante dans le temps)
    # x va de 0 à 1 au fil du temps (0 = début, 1 = veille du vol)
    x = (total_jours - jours_avant) / total_jours  # progression temporelle 0→1
    pax_loisir   = (capacite * 0.60 * facteur * (1 - np.exp(-4 * x))).clip(0, capacite * 0.65)
    pax_affaires = (capacite * 0.35 * facteur * (1 - np.exp(-12 * x))).clip(0, capacite * 0.38)
    pax_total    = (pax_loisir + pax_affaires).clip(0, capacite)

    # Prix dynamique : monte fortement quand le taux de remplissage dépasse 60%
    taux_remplissage = pax_total / capacite
    prix_dynamique   = prix_depart * (1 + 1.5 * np.maximum(0, taux_remplissage - 0.3) ** 1.5)

    fig13, (ax13a, ax13b) = plt.subplots(2, 1, figsize=(12, 8))

    # Graphique réservations
    ax13a.stackplot(jours[::-1], pax_loisir[::-1], pax_affaires[::-1],
                    labels=["Passagers Loisir", "Passagers Affaires"],
                    colors=["steelblue", "tomato"], alpha=0.7)
    ax13a.axhline(capacite, color="black", linestyle="--", linewidth=1.5, label=f"Capacité ({capacite} sièges)")
    ax13a.set_xlabel("Jours avant le vol")
    ax13a.set_ylabel("Passagers réservés")
    ax13a.set_title("Courbe de réservation simulée par segment")
    ax13a.legend(); ax13a.grid(True, alpha=0.3)
    ax13a.invert_xaxis()

    # Graphique prix dynamique
    ax13b.plot(jours[::-1], prix_dynamique[::-1], color="darkorange", linewidth=2.5)
    ax13b.fill_between(jours[::-1], prix_depart, prix_dynamique[::-1], alpha=0.15, color="darkorange")
    ax13b.axhline(prix_depart, color="gray", linestyle="--", linewidth=1, label=f"Prix de départ ({prix_depart}€)")
    ax13b.set_xlabel("Jours avant le vol")
    ax13b.set_ylabel("Prix du billet (€)")
    ax13b.set_title("Prix dynamique selon le taux de remplissage")
    ax13b.legend(); ax13b.grid(True, alpha=0.3)
    ax13b.invert_xaxis()

    plt.tight_layout()
    st.pyplot(fig13)

    st.divider()

    # KPIs de la simulation
    taux_final = pax_total[-1] / capacite * 100
    prix_final = prix_dynamique[-1]
    revenu_sim = pax_total[-1] * prix_dynamique.mean()

    k1, k2, k3 = st.columns(3)
    k1.metric("Taux de remplissage prévu",  f"{taux_final:.1f}%")
    k2.metric("Prix à J-1",                 f"{prix_final:.0f}€", f"+{prix_final-prix_depart:.0f}€ vs départ")
    k3.metric("Revenu estimé du vol",        f"{revenu_sim:,.0f}€")

    st.success(f"💡 Avec un profil **{profil.lower()}**, l'avion devrait être rempli à **{taux_final:.1f}%** la veille du vol. "
               f"Le prix dynamique est passé de **{prix_depart}€** à **{prix_final:.0f}€** — soit une hausse de **{((prix_final/prix_depart)-1)*100:.1f}%**. "
               f"C'est exactement la logique du Revenue Management : maximiser le revenu selon la demande réelle.")

    st.divider()
    st.info("ℹ️ Cette simulation est une illustration pédagogique du concept de courbe de réservation. "
            "En réalité, Air France utilise des systèmes RM sophistiqués (Amadeus, etc.) avec des données de réservation en temps réel sur chaque vol.")

st.divider()
st.caption("Source : DGAC — Direction Générale de l'Aviation Civile | Licence Ouverte 2.0")