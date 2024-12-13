import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Carregar dados do arquivo CSV
def carregar_dados():
    return pd.read_csv("pacientes.csv")

# Treinamento do modelo
def treinar_modelo():
    df = carregar_dados()
    X = df[["Glicemia", "Temperatura", "Sintomas"]]
    y = df["Gravidade"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    return modelo, scaler, df
