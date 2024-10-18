# utils.py

import numpy as np
from sklearn.metrics import cohen_kappa_score
import wandb
import pandas as pd

import lightgbm as lgb
import xgboost as xgb

def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")

def threshold_Rounder(oof_non_rounded, thresholds):
    return np.where(
        oof_non_rounded < thresholds[0],
        0,
        np.where(
            oof_non_rounded < thresholds[1],
            1,
            np.where(oof_non_rounded < thresholds[2], 2, 3),
        ),
    )

def evaluate_predictions(thresholds, y_true, oof_non_rounded):
    rounded_p = threshold_Rounder(oof_non_rounded, thresholds)
    return -quadratic_weighted_kappa(y_true, rounded_p)

def optimize_predictions(y, oof_non_rounded):
    from scipy.optimize import minimize
    KappaOptimizer = minimize(
        evaluate_predictions,
        x0=[0.5, 1.5, 2.5],
        args=(y, oof_non_rounded),
        method="Nelder-Mead",
    )
    return KappaOptimizer

def _apply_thresholds(predictions, thresholds):
    return np.digitize(predictions, bins=np.sort(thresholds))

def log_features_to_wandb(features):
    features_text = "\n".join(features)
    wandb.log({"features": wandb.Html(f"<pre>{features_text}</pre>")})

def save_model_to_wandb(model, model_name, models_dir):
    import pickle
    model_path = f"{models_dir}/{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    artifact = wandb.Artifact(name=model_name, type="model")
    artifact.add_file(model_path)
    wandb.run.log_artifact(artifact)

def log_feature_importance_to_wandb(model):
    if hasattr(model.model, 'feature_importances_'):
        # LightGBMの場合
        if isinstance(model.model, lgb.LGBMRegressor):
            feature_names = model.model.booster_.feature_name()
            feature_importances = model.model.feature_importances_
        # XGBoostの場合
        elif isinstance(model.model, xgb.XGBRegressor):
            feature_names = model.model.get_booster().feature_names
            feature_importances = model.model.feature_importances_
        else:
            raise ValueError("Model type not supported for feature importance logging.")

        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        })
        wandb.log({"feature_importance": wandb.Table(dataframe=feature_importance_df)})

