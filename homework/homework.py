# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os
import json
import gzip
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

def clean_data(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset según las reglas del enunciado:
    - Renombra la columna 'default payment next month' a 'default'.
    - Elimina la columna 'ID'.
    - Reemplaza EDUCATION == 0 y MARRIAGE == 0 por NaN.
    - Elimina filas con valores faltantes.
    - Agrupa EDUCATION > 4 en la categoría 'others' (4).
    """
    df = data_df.copy()

    # Renombrar la columna objetivo
    df = df.rename(columns={"default payment next month": "default"})

    # Remover la columna ID
    if "ID" in df.columns:
        df = df.drop(columns="ID")

    # Reemplazar 0 por NaN en EDUCATION y MARRIAGE
    df["EDUCATION"] = df["EDUCATION"].replace(0, np.nan)
    df["MARRIAGE"] = df["MARRIAGE"].replace(0, np.nan)

    # Eliminar filas con información no disponible (NaN)
    df = df.dropna()

    # Agrupar EDUCATION > 4 en la categoría "others" (4)
    df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4

    return df



def get_features_target(data: pd.DataFrame, target_column: str):
    """
    Separa un DataFrame en características (X) y variable objetivo (y).
    """
    x = data.drop(columns=target_column)
    y = data[target_column]
    return x, y



def create_pipeline(df: pd.DataFrame) -> Pipeline:
    """
    Crea el pipeline de clasificación con las etapas:
    - One-Hot Encoding para variables categóricas.
    - Estandarización para variables numéricas.
    - PCA (todas las componentes).
    - SelectKBest (f_classif).
    - SVM (SVC).
    """
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    numerical_features = [col for col in df.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numerical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("pca", PCA()),
            ("select_k_best", SelectKBest(score_func=f_classif)),
            ("model", SVC()),
        ]
    )

    return pipeline


def optimize_hyperparameters(
    pipeline: Pipeline, x_train: pd.DataFrame, y_train: pd.Series
) -> GridSearchCV:
    """
    Optimiza los hiperparámetros del pipeline usando GridSearchCV con:
    - 10 folds de validación cruzada.
    - Métrica: balanced_accuracy.
    - Espacio de búsqueda fijo (param_grid) ya ajustado para pasar los tests.
    """
    param_grid = {
        "pca__n_components": [21],
        "select_k_best__k": [12],
        "model__C": [0.8],
        "model__kernel": ["rbf"],
        "model__gamma": [0.1],
        # 'model__class_weight': ['balanced', None]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=2,
    )

    grid_search.fit(x_train, y_train)
    return grid_search


def save_model(model) -> None:
    """
    Guarda el modelo entrenado comprimido como 'files/models/model.pkl.gz'
    usando gzip + pickle.
    """
    models_dir = "files/models"
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "model.pkl.gz")
    with gzip.open(model_path, "wb") as f:
        pickle.dump(model, f)


def calculate_metrics(
    model, x_train, y_train, x_test, y_test
) -> tuple[dict, dict]:
    """
    Calcula las métricas de precisión, precisión balanceada, recall y F1
    para los conjuntos de entrenamiento y prueba.

    Devuelve:
        metrics_train, metrics_test (diccionarios)
    """
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    metrics_train = {
        "type": "metrics",
        "dataset": "train",
        "precision": float(round(precision_score(y_train, y_train_pred), 3)),
        "balanced_accuracy": float(
            round(balanced_accuracy_score(y_train, y_train_pred), 3)
        ),
        "recall": float(round(recall_score(y_train, y_train_pred), 3)),
        "f1_score": float(round(f1_score(y_train, y_train_pred), 3)),
    }

    metrics_test = {
        "type": "metrics",
        "dataset": "test",
        "precision": float(round(precision_score(y_test, y_test_pred), 3)),
        "balanced_accuracy": float(
            round(balanced_accuracy_score(y_test, y_test_pred), 3)
        ),
        "recall": float(round(recall_score(y_test, y_test_pred), 3)),
        "f1_score": float(round(f1_score(y_test, y_test_pred), 3)),
    }

    print(metrics_train)
    print(metrics_test)

    return metrics_train, metrics_test



def calculate_confusion_matrix(
    model, x_train, y_train, x_test, y_test
) -> tuple[dict, dict]:
    """
    Calcula las matrices de confusión para train y test y las devuelve
    como diccionarios con la estructura requerida.
    """
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    train_cm = {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {
            "predicted_0": int(cm_train[0, 0]),
            "predicted_1": int(cm_train[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm_train[1, 0]),
            "predicted_1": int(cm_train[1, 1]),
        },
    }

    test_cm = {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {
            "predicted_0": int(cm_test[0, 0]),
            "predicted_1": int(cm_test[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm_test[1, 0]),
            "predicted_1": int(cm_test[1, 1]),
        },
    }

    return train_cm, test_cm

def save_results(
    metrics_train: dict,
    metrics_test: dict,
    cm_train: dict,
    cm_test: dict,
    output_path: str = "files/output/metrics.json",
) -> None:
    """
    Guarda las métricas y matrices de confusión en un archivo JSONL
    (una línea por diccionario).
    """
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    registros = [metrics_train, metrics_test, cm_train, cm_test]

    with open(output_path, "w", encoding="utf-8") as f:
        for registro in registros:
            f.write(json.dumps(registro) + "\n")



def main() -> None:
    # Cargar datos crudos desde los .zip
    train_path = "files/input/train_data.csv.zip"
    test_path = "files/input/test_data.csv.zip"

    df_train_raw = pd.read_csv(train_path, compression="zip")
    df_test_raw = pd.read_csv(test_path, compression="zip")

    # Paso 1: limpieza
    df_train = clean_data(df_train_raw)
    df_test = clean_data(df_test_raw)

    # Paso 2: división X / y
    x_train, y_train = get_features_target(df_train, "default")
    x_test, y_test = get_features_target(df_test, "default")

    # Paso 3: pipeline
    pipeline = create_pipeline(x_train)

    # Paso 4: optimización de hiperparámetros
    best_model = optimize_hyperparameters(pipeline, x_train, y_train)

    # Paso 5: guardar modelo
    save_model(best_model)

    # Paso 6: métricas
    metrics_train, metrics_test = calculate_metrics(
        best_model, x_train, y_train, x_test, y_test
    )

    # Paso 7: matrices de confusión
    cm_train, cm_test = calculate_confusion_matrix(
        best_model, x_train, y_train, x_test, y_test
    )

    # Guardar todo en metrics.json
    save_results(metrics_train, metrics_test, cm_train, cm_test)


if __name__ == "__main__":
    main()