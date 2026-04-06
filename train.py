"""
=============================================================================
MÓDULO DE ENTRENAMIENTO DEL MODELO
=============================================================================
Entrena LightGBM, evalúa métricas y exporta el modelo como .pkl.
=============================================================================
"""

import numpy as np
import pandas as pd
import pickle
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
import config
from preprocessing import (
    run_preprocessing_pipeline,
    prepare_features,
    save_encoders,
)


def split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Divide los datos en train y test."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )
    print(f"[INFO] Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")
    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> lgb.LGBMRegressor:
    """Entrena el modelo LightGBM con early stopping."""
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO DEL MODELO")
    print("=" * 60)

    # Identificar columnas categóricas (ya encodeadas como int)
    cat_cols = [c for c in config.CATEGORICAL_FEATURES if c in X_train.columns]

    model = lgb.LGBMRegressor(**config.LGBM_PARAMS)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=100),
        ],
        categorical_feature=cat_cols,
    )

    best_iter = model.best_iteration_
    print(f"\n[INFO] Mejor iteración: {best_iter}")

    return model


def evaluate_model_from_predictions(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evalúa métricas a partir de arrays de valores reales y predichos."""
    metrics = {
        "MAE": mean_absolute_error(y_real, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_real, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_real, y_pred) * 100,
        "R2": r2_score(y_real, y_pred),
        "Median_AE": np.median(np.abs(y_real - y_pred)),
    }

    print("\n" + "=" * 60)
    print("MÉTRICAS DE EVALUACIÓN (escala original CLP)")
    print("=" * 60)
    print(f"  MAE  (Error Absoluto Medio):        ${metrics['MAE']:,.0f}")
    print(f"  RMSE (Raíz Error Cuadrático Medio):  ${metrics['RMSE']:,.0f}")
    print(f"  MAPE (Error Porcentual Medio):        {metrics['MAPE']:.2f}%")
    print(f"  R²   (Coef. Determinación):           {metrics['R2']:.4f}")
    print(f"  Median AE (Error Mediano):            ${metrics['Median_AE']:,.0f}")

    print("\n" + "-" * 60)
    print("INTERPRETACIÓN:")
    if metrics["R2"] >= 0.90:
        print("  R² >= 0.90: El modelo explica muy bien la variabilidad del precio.")
    elif metrics["R2"] >= 0.80:
        print("  R² >= 0.80: Buen modelo, explica la mayoría de la variabilidad.")
    elif metrics["R2"] >= 0.70:
        print("  R² >= 0.70: Modelo aceptable, hay margen de mejora.")
    else:
        print("  R² < 0.70: Modelo débil. Revisar features o datos.")

    if metrics["MAPE"] <= 10:
        print("  MAPE <= 10%: Predicciones muy precisas en promedio.")
    elif metrics["MAPE"] <= 20:
        print("  MAPE <= 20%: Precisión aceptable para precios de autos.")
    else:
        print("  MAPE > 20%: Error promedio alto, considerar más features.")

    return metrics


def analyze_predictions_from_arrays(y_real: np.ndarray, y_pred: np.ndarray, n_samples: int = 10):
    """Muestra ejemplos de predicciones vs valores reales."""
    print("\n" + "=" * 60)
    print(f"EJEMPLOS DE PREDICCIONES (primeras {n_samples})")
    print("=" * 60)
    print(f"  {'Real':>15s} | {'Predicho':>15s} | {'Diferencia':>15s} | {'Error %':>8s}")
    print("  " + "-" * 62)

    indices = np.random.RandomState(42).choice(len(y_real), size=min(n_samples, len(y_real)), replace=False)

    for idx in indices:
        real = y_real.iloc[idx] if hasattr(y_real, 'iloc') else y_real[idx]
        pred = y_pred[idx]
        diff = pred - real
        pct = abs(diff) / real * 100
        print(f"  ${real:>13,.0f} | ${pred:>13,.0f} | ${diff:>13,.0f} | {pct:>6.1f}%")


def get_feature_importance(
    model: lgb.LGBMRegressor,
    feature_names: list,
    top_n: int = 15,
) -> pd.DataFrame:
    """Muestra la importancia de cada feature."""
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n" + "=" * 60)
    print(f"TOP {top_n} FEATURES MÁS IMPORTANTES")
    print("=" * 60)
    for i, row in importance.head(top_n).iterrows():
        bar = "█" * int(row["importance"] / importance["importance"].max() * 30)
        print(f"  {row['feature']:20s} | {row['importance']:6.0f} | {bar}")

    return importance


def save_model(model: lgb.LGBMRegressor):
    """Exporta el modelo como .pkl."""
    os.makedirs(os.path.dirname(config.MODEL_OUTPUT_PATH), exist_ok=True)

    with open(config.MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(model, f)

    size_mb = os.path.getsize(config.MODEL_OUTPUT_PATH) / (1024 * 1024)
    print(f"\n[INFO] Modelo guardado en: {config.MODEL_OUTPUT_PATH} ({size_mb:.2f} MB)")


def run_training_pipeline():
    """Pipeline completo: preprocesamiento → entrenamiento → evaluación → exportación."""
    print("=" * 60)
    print("PIPELINE DE ENTRENAMIENTO - PREDICTOR DE PRECIOS DE AUTOS")
    print("=" * 60)

    # 1. Preprocesamiento
    df, encoders = run_preprocessing_pipeline()

    # 2. Preparar features
    X, y = prepare_features(df)

    # 3. Transformar target a log (precios tienen distribución log-normal)
    y_log = np.log1p(y)

    # 4. Split
    X_train, X_test, y_train_log, y_test_log = split_data(X, y_log)

    # 5. Entrenar (en escala log)
    model = train_model(X_train, y_train_log, X_test, y_test_log)

    # 6. Evaluar (reconvertir a escala original para métricas interpretables)
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test_log)
    metrics = evaluate_model_from_predictions(y_test_real, y_pred)

    # 7. Feature importance
    importance = get_feature_importance(model, list(X.columns))

    # 8. Ejemplos de predicciones
    analyze_predictions_from_arrays(y_test_real, y_pred)

    # 9. Guardar modelo y artefactos
    save_model(model)
    save_encoders(encoders, list(X.columns))

    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    print(f"  Modelo: {config.MODEL_OUTPUT_PATH}")
    print(f"  Encoders: {config.ENCODERS_OUTPUT_PATH}")
    print(f"  Features: {config.FEATURE_NAMES_PATH}")
    print(f"  R²: {metrics['R2']:.4f} | MAPE: {metrics['MAPE']:.2f}%")

    return model, metrics


if __name__ == "__main__":
    model, metrics = run_training_pipeline()
