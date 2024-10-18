# training.py

import numpy as np
from sklearn.model_selection import StratifiedKFold
from utils import quadratic_weighted_kappa, optimize_predictions, _apply_thresholds
from utils import log_features_to_wandb, save_model_to_wandb, log_feature_importance_to_wandb
import wandb
import pandas as pd

def prepare_data(train, test):
    X = train.drop(["sii"], axis=1)
    y = train["sii"]
    return X, y

def train_fold(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    return model, y_train_pred, y_val_pred

def evaluate_fold(y_train, y_train_pred, y_val, y_val_pred):
    train_kappa = quadratic_weighted_kappa(y_train, np.round(y_train_pred).astype(int))
    val_kappa = quadratic_weighted_kappa(y_val, np.round(y_val_pred).astype(int))
    return train_kappa, val_kappa

def train_model(ModelClass, model_params, train, test, sample, features, n_splits=5, seed=42):
    X, y = prepare_data(train, test)
    SKF = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_non_rounded = np.zeros(len(y), dtype=float)
    oof_rounded = np.zeros(len(y), dtype=int)
    test_preds = np.zeros((len(test), n_splits))

    # 訓練されたモデルを保存するリストを定義
    models = []
    train_scores = []
    val_scores = []

    for fold, (train_idx, val_idx) in enumerate(SKF.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # モデルの定義
        if ModelClass.__name__ == 'NeuralNetworkModel':
            model_params['input_size'] = X_train.shape[1]
        model = ModelClass(model_params)

        # このフォールドでモデルを訓練し、予測を取得
        model, y_train_pred, y_val_pred = train_fold(
            model, X_train, y_train, X_val, y_val
        )

        # モデルを保存
        models.append(model)

        # このフォールドの性能を評価
        train_kappa, val_kappa = evaluate_fold(
            y_train, y_train_pred, y_val, y_val_pred
        )
        train_scores.append(train_kappa)
        val_scores.append(val_kappa)

        # 予測を保存
        oof_non_rounded[val_idx] = y_val_pred
        oof_rounded[val_idx] = np.round(y_val_pred).astype(int)
        test_pred = model.predict(test)
        test_preds[:, fold] = test_pred

        print(f"Fold {fold+1}: Train Kappa={train_kappa}, Validation Kappa={val_kappa}")
        wandb.log(
            {"Fold": fold + 1, "Train QWK": train_kappa, "Validation QWK": val_kappa}
        )

    # 全体の性能を表示
    mean_train_kappa = np.mean(train_scores)
    mean_val_kappa = np.mean(val_scores)
    print(f"Mean Train Kappa: {mean_train_kappa}")
    print(f"Mean Validation Kappa: {mean_val_kappa}")

    wandb.log({"Mean Train QWK": mean_train_kappa, "Mean Validation QWK": mean_val_kappa})

    # 予測を最適化
    KappaOptimizer = optimize_predictions(y, oof_non_rounded)
    best_thresholds = KappaOptimizer.x

    # フォールド間の予測を平均化
    test_preds_mean = test_preds.mean(axis=1)

    # 最適化された閾値を適用
    final_test_preds = _apply_thresholds(test_preds_mean, best_thresholds)

    # 提出用データフレームを作成
    submission = pd.DataFrame({"id": sample["id"], "sii": final_test_preds})

    return submission, models, mean_val_kappa
