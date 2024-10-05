refered from: https://upura.hatenablog.com/entry/2018/12/28/225234

# Structures
```
.
├── configs
│   └── default.json
├── data
│   ├── input
│   │   ├── sample_submission.csv
│   │   ├── train.csv
│   │   └── test.csv
│   └── output
├── features
│   ├── __init__.py
│   ├── base.py
│   └── create.py
├── logs
│   └── logger.py
├── models
│   └── lgbm.py
├── notebooks
│   └── eda.ipynb
├── scripts
│   └── convert_to_feather.py
├── utils
│   └── __init__.py
├── .gitignore
├── .pylintrc
├── LICENSE
├── README.md
├── run.py
└── tox.ini
```
# Commands

## Change data to feather format

```
python scripts/convert_to_feather.py
```

## Create features

```
python features/create.py
```

## Run LightGBM

```
python run.py
```

## flake8

```
flake8 .
```

# Usage

## features/base.py
このコードは、Pythonを使用して特徴量抽出とファイルの管理を行うための一般的な構造を提供します。以下に、各部分の簡単な説明を示します。

### インポートセクション

- **`argparse`**: コマンドライン引数を解析するためのライブラリ。
- **`inspect`**: オブジェクトのメタデータ情報を取得するためのモジュール。
- **`re`**: 正規表現を使用するためのライブラリ。
- **`ABCMeta` と `abstractmethod`**: 抽象基底クラスを定義するためのモジュール。
- **`Path`**: ファイルシステムパスを操作するためのモジュール。
- **`pandas`**: データ操作と分析のためのライブラリ。
- **`time`**: 時間を計測するためのモジュール。
- **`contextmanager`**: コンテキストマネージャーを簡単に作成するためのデコレータ。

### `timer` 関数

コンテキストマネージャーとしての `timer` は、指定された名前で処理の開始と終了にかかった時間をコンソールに表示します。特に処理時間の計測に便利です。

### `get_arguments` 関数

コマンドラインからのオプション `--force` または `-f` を解析し、既存のファイルを上書きするかどうかを制御する機能を提供します。戻り値は解析された引数が格納されたオブジェクトです。

### `get_features` 関数

指定された名前空間内のすべての `Feature` クラスを反復処理し、それらのインスタンスを生成します。`Feature` クラスのサブクラスを見つけるために用いられます。

### `generate_features` 関数

指定した名前空間にある各 `Feature` クラスについて、特徴量を生成します。もし `overwrite` フラグが `True` でない限り既存の特徴量ファイルが存在する場合、その特徴量の生成をスキップします。

### `Feature` クラス

この抽象基底クラスは、特徴量を生成する際の基本的な構造を提供し、サブクラス化されることを目的としています。

- **コンストラクタ (__init__)**: クラス名から自動的に特徴量名を生成します。また、`train` と `test` のデータフレームを初期化し、それぞれの保存先パスを設定します。

- **`run` メソッド**: 特徴量の生成プロセスを実行しながらタイマーで時間を計測します。

- **`create_features` メソッド**: 各特徴量に固有の処理を記述するために、サブクラスでオーバーライドされる必要があります。

- **`save` メソッド**: 特徴量をファイルに保存します。

- **`load` メソッド**: 保存された特徴量を読み込みます。

このコード全体は柔軟に拡張可能で、様々な種類のデータに対する特徴量生成と管理を効率的に扱う設計となっています。
