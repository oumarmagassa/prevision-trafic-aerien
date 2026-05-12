# Prévision du trafic aérien

## Objectif du projet

Ce projet a pour objectif de prévoir le trafic passagers à partir de données publiques, en comparant plusieurs méthodes de prévision statistique et de machine learning.

## Méthodes utilisées

- Analyse exploratoire des données
- Séries temporelles
- Modèle ARIMA
- Modèle XGBoost
- Validation croisée temporelle
- Évaluation avec RMSE et MAE

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- matplotlib
- Streamlit
## Lancer le dashboard

```bash
streamlit run app/dashboard.py

```

## Structure du projet

```text
prevision-trafic-aerien/
├── data/
├── notebooks/
├── src/
├── figures/
├── app/
├── README.md
```
## État d’avancement

- [x] Structure du projet
- [x] Jeu de données initial
- [x] Analyse exploratoire
- [x] Visualisation du trafic
- [x] Modèle naïf de référence
- [x] Premier modèle XGBoost
- [x] Dashboard Streamlit initial
- [ ] Ajout de données réelles plus complètes
- [ ] Validation croisée temporelle
- [ ] Comparaison ARIMA / XGBoost
- [ ] Amélioration du dashboard
