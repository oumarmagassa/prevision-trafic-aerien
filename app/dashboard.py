import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.title("Prévision du trafic aérien")

st.write(
    "Ce dashboard présente une première analyse du trafic passagers "
    "et servira ensuite à comparer les modèles de prévision."
)

df = pd.read_csv("data/trafic_aerien.csv")
df["date"] = pd.to_datetime(df["date"])

st.subheader("Aperçu des données")
st.dataframe(df)

st.subheader("Évolution du trafic passagers")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["date"], df["passagers"], marker="o")
ax.set_xlabel("Date")
ax.set_ylabel("Nombre de passagers")
ax.set_title("Évolution du trafic passagers")
ax.grid(True)

st.pyplot(fig)

st.subheader("Statistiques descriptives")
st.write(df["passagers"].describe())
