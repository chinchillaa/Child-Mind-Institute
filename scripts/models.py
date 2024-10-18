# models.py

import lightgbm as lgb
import xgboost as xgb
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim

class BaseModel(ABC):
    @abstractmethod
    def create_model(self, params):
        pass

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X):
        pass

class LightGBMModel(BaseModel):
    def __init__(self, params):
        self.model = self.create_model(params)

    def create_model(self, params):
        return lgb.LGBMRegressor(**params)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

class XGBoostModel(BaseModel):
    def __init__(self, params):
        self.model = self.create_model(params)

    def create_model(self, params):
        return xgb.XGBRegressor(**params)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

class NeuralNetworkModel(BaseModel):
    def __init__(self, params):
        self.params = params
        self.model = self.create_model(params)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=params['learning_rate'])

    def create_model(self, params):
        input_size = params['input_size']
        hidden_size = params['hidden_size']
        output_size = 1
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def fit(self, X_train, y_train):
        X_train = torch.from_numpy(X_train.values).float()
        y_train = torch.from_numpy(y_train.values).float().unsqueeze(1)
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=True)
        
        self.model.train()
        for epoch in range(self.params['epochs']):
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        X = torch.from_numpy(X.values).float()
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
        return outputs.numpy().flatten()
