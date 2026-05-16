# ✈️ Prévision du trafic passagers — Air France

> Projet de data science appliqué au **Revenue Management aérien**.  
> Analyse et prévision du trafic passagers Air France sur 15 ans (2010–2024) à partir de données officielles.

---

## 🎯 Objectif

Ce projet simule le travail d'un **Analyste de vols - Revenue Management** :

- Comprendre la structure du trafic (saisonnalité, tendances, anomalies)
- Mesurer l'impact du Covid-19 et analyser la reprise
- Construire et comparer des modèles de prévision (ARIMA, XGBoost)
- Prévoir le trafic 2025–2026 pour anticiper les décisions tarifaires
- Simuler l'impact financier des ajustements de prix
- Modéliser les courbes de réservation par segment de clientèle

---

## 🌐 Dashboard en ligne

👉 **[Accéder au dashboard](https://prevision-trafic-aerien.streamlit.app)**

---

## 📊 Source des données

**Direction Générale de l'Aviation Civile (DGAC)** — [data.gouv.fr](https://www.data.gouv.fr/datasets/trafic-aerien-commercial-mensuel-francais-par-paire-daeroports-par-sens-depuis-1990)

- Données officielles et publiques (Licence Ouverte 2.0)
- Série `ASP_CIE` : trafic mensuel par compagnie aérienne
- Filtrées sur **Air France** uniquement
- **180 mois** de données : janvier 2010 → décembre 2024

| Variable | Description |
|----------|-------------|
| `passagers` | Nombre de passagers transportés |
| `vols` | Nombre de vols effectués |
| `passagers_km` | Passagers-kilomètres (milliards) |

---

## 🔍 Analyses réalisées

### 1. Analyse exploratoire
- Évolution du trafic sur 15 ans
- Saisonnalité mensuelle (pic juillet, creux février)
- Comparaison année par année avec année de référence paramétrable
- Matrice de corrélations pour la sélection des features

### 2. Impact Covid & reprise
- Creux historique : **avril 2020 à 2% du niveau normal**
- Premier mois à 95% du niveau pré-Covid : **mai 2023** (3 ans de reprise)
- Niveau moyen 2024 : **~91% du niveau pré-Covid**

### 3. Modèles de prévision

| Modèle | MAE | RMSE | MAPE |
|--------|-----|------|------|
| Naïf (même mois -1 an) | ~300K | ~380K | ~9% |
| ARIMA (1,1,1) | 355K | 401K | 10.2% |
| **XGBoost** ✅ | **106K** | **133K** | **3.1%** |

**XGBoost** gagne grâce aux variables de lags (lag_1, lag_3, lag_12) et à la saisonnalité.

### 4. Prévisions 2025–2026
- Prévisions mois par mois avec horizon paramétrable (6, 12, 18, 24 mois)
- Croissance stable prévue de ~+2.5% vs 2024

### 5. Rendement & Load Factor
- Distance moyenne par passager : ~3 050 km (profil long-courrier)
- Passagers par vol : proxy du taux de remplissage
- Saisonnalité du rendement : pic en juillet–août

### 6. Recommandations tarifaires
- Décisions automatiques basées sur les prévisions vs l'historique
- Seuil de décision paramétrable (1% à 15%)
- 3 niveaux : hausse tarifaire / maintien / stimulation par promotions

### 7. Simulation de revenus
- Impact financier en millions d'euros des ajustements tarifaires
- Sliders interactifs : % hausse haute saison, % baisse basse saison, élasticité
- Comparaison revenu de base vs revenu simulé mois par mois

### 8. Courbes de réservation
- Simulation du comportement de réservation par segment (Loisir / Affaires)
- Prix dynamique qui monte selon le taux de remplissage
- Paramètres interactifs : capacité, profil de demande, horizon, prix de départ

---

## 🗂️ Structure du projet

```
prevision-trafic-aerien/
├── data/
│   └── trafic_airfrance.csv       # Données DGAC filtrées Air France
├── notebooks/
│   └── exploration_donnees.ipynb  # Analyse exploratoire complète
├── src/
│   ├── preparation_donnees.py     # Chargement et nettoyage
│   ├── modele_baseline.py         # Modèle naïf de référence
│   └── modele_xgboost.py          # Modèle XGBoost avec features temporelles
├── app/
│   └── dashboard.py               # Dashboard Streamlit (9 onglets)
├── figures/                       # Graphiques générés
├── requirements.txt
├── COMMANDES.md
└── README.md
```

---

## 🚀 Lancer le projet

### Installation

```bash
git clone https://github.com/oumarmagassa/prevision-trafic-aerien.git
cd prevision-trafic-aerien
pip install -r requirements.txt
```

### Dashboard interactif

```bash
python -m streamlit run app/dashboard.py
```

### Scripts individuels

```bash
python src/modele_baseline.py
python src/modele_xgboost.py
jupyter notebook notebooks/exploration_donnees.ipynb
```

---

## 🛠️ Technologies

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)
![ARIMA](https://img.shields.io/badge/ARIMA-TimeSeries-green)

- **pandas / numpy** — manipulation des données
- **XGBoost** — modèle de machine learning
- **statsmodels** — modèle ARIMA
- **scikit-learn** — métriques d'évaluation
- **matplotlib** — visualisations
- **Streamlit** — dashboard interactif

---

## 💡 Insights Revenue Management

1. **Saisonnalité forte et prévisible** → juillet +25% vs moyenne annuelle
2. **Reprise post-Covid incomplète** → opportunité de stimulation tarifaire en basse saison
3. **XGBoost fiable à 96.9%** (MAPE 3.1%) → utilisable pour des décisions opérationnelles
4. **Simulation revenus** → une hausse de 10% en haute saison génère plusieurs millions d'euros
5. **Courbes de réservation** → les affaires réservent tard, les loisirs tôt → tarification différenciée

---

## 👤 Auteur

**Oumar Magassa**  
Projet Data Science — Prévision & Optimisation du Revenue Management Aérien
Données : DGAC — Licence Ouverte 2.0