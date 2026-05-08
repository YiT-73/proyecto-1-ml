# ===============================
# CASO B: PREDICCIÓN DE CHURN
# ===============================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Semilla para replicabilidad
SEED = 42


df = pd.read_csv("test.csv")


print(df.head())
print(df.info())
print(df.isnull().sum())


# Convertir TotalCharges a numérico
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Eliminar filas con valores nulos
df = df.dropna()

# Eliminar customerID porque no aporta al modelo
df = df.drop("customerID", axis=1)

print(df.columns.tolist())

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop("Churn", axis=1)
y = df["Churn"]



numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

print("Variables numéricas:", numeric_features)
print("Variables categóricas:", categorical_features)



preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

models = {
    "Regresión Logística": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED
    ),
    "Árbol de Decisión": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=SEED
    )
}

results = []

for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results.append({
        "Modelo": name,
        "Accuracy": acc,
        "F1-score": f1
    })

    print("\n====================")
    print(name)
    print("====================")
    print("Accuracy:", acc)
    print("F1-score:", f1)
    print("Matriz de confusión:")
    print(cm)
    print(classification_report(y_test, y_pred))

    results_df = pd.DataFrame(results)
    print(results_df)
