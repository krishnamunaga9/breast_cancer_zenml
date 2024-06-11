from zenml import pipeline
from zenml import step
from data.load_data import load_data
from features.feature_engineering import preprocess_data
from models.model import train_model, evaluate_model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier

@step
def data_loader() -> pd.DataFrame:
    return load_data()

@step
def data_preprocessor(df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
    return preprocess_data(df)

@step
def model_trainer(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    return train_model(X_train, y_train)

@step
def model_evaluator(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, str]:
    return evaluate_model(model, X_test, y_test)

@pipeline
def training_pipeline(data_loader, data_preprocessor, model_trainer, model_evaluator):
    df = data_loader()
    preprocessed_data = data_preprocessor(df)
    X, y = preprocessed_data.artifacts  # Accessing the artifacts attribute of the StepArtifact object
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = model_trainer(X_train, y_train)
    evaluation = model_evaluator(model, X_test, y_test)
    accuracy, report = evaluation.artifacts
    
    print(f'Accuracy: {accuracy}')
    print(report)
