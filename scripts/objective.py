# objective.py

import wandb
from models import LightGBMModel, XGBoostModel, NeuralNetworkModel
from training import train_model
from utils import log_features_to_wandb, save_model_to_wandb, log_feature_importance_to_wandb

def objective(trial, train, test, sample, features, search_space, models_dir, output_dir):
    import datetime
    # ハイパーパラメータのサンプリング
    model_type = search_space.model_type
    param_grid = {}

    for param in search_space.int_params:
        param_grid[param.name] = trial.suggest_int(param.name, param.low, param.high)
    for param in search_space.float_params:
        if param.name in ['learning_rate', 'reg_alpha', 'reg_lambda']:
            param_grid[param.name] = trial.suggest_loguniform(param.name, param.low, param.high)
        else:
            param_grid[param.name] = trial.suggest_uniform(param.name, param.low, param.high)
    for param in search_space.categorical_params:
        param_grid[param.name] = trial.suggest_categorical(param.name, param.choices)

    # モデルの選択
    if model_type == 'lightgbm':
        ModelClass = LightGBMModel
    elif model_type == 'xgboost':
        ModelClass = XGBoostModel
    elif model_type == 'neural_network':
        ModelClass = NeuralNetworkModel
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # WandBの設定
    wandb_kwargs = {"project": "cmi_optuna_sweep"}
    wandbc = wandb.init(project="cmi_optuna_sweep", config=param_grid, reinit=True)

    with wandbc:
        # 特徴量をログに保存
        log_features_to_wandb(features)

        # モデルの訓練
        Submission, models, mean_val_kappa = train_model(ModelClass, param_grid, train, test, sample, features)

        # モデルを保存（最初のモデルのみ）
        save_model_to_wandb(models[0], model_name=f"{model_type}_model_trial_{trial.number}_fold_0", models_dir=models_dir)

        # 特徴量重要度をログに保存（LightGBMとXGBoostのみ対応）
        if model_type in ['lightgbm', 'xgboost']:
            log_feature_importance_to_wandb(models[0])

        # 提出用ファイルを保存
        Submission.to_csv(f"{output_dir}/submission_trial_{trial.number}.csv", index=False)

    # 目的関数の値（ここでは検証データでの平均QWKの負値を最小化）
    return -mean_val_kappa
