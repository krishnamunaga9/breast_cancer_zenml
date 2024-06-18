from zenml import pipeline, step
from data.load_data import load_data
from features.feature_engineering import preprocess_data
from models.model import train_model, evaluate_model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from typing import Tuple, Annotated
from sklearn.ensemble import RandomForestClassifier

@step
def data_loader() -> Tuple[Annotated[pd.DataFrame, "df"]]:
    df = load_data()
    return df

@step
def data_preprocessor(df: pd.DataFrame) -> Tuple[Annotated[pd.DataFrame, "X_train"],
      Annotated[pd.DataFrame, "y_train"],
      Annotated[pd.Series, "X_test"],
      Annotated[pd.Series, "y_test"]]:
    X_train, X_test, y_train, y_test = preprocess_data(df)
    return X_train, X_test, y_train, y_test

@step
def model_trainer(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    return train_model(X_train, y_train)

@step
def model_evaluator(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Annotated[float, "accuracy"],
      Annotated[str, "report"]]:
    accuracy, report = evaluate_model(model, X_test, y_test)
    return accuracy, report

@pipeline
def training_pipeline():
    df = data_loader()
    #preprocessed_data = data_preprocessor(df)
    X_train, X_test, y_train, y_test = data_preprocessor(df)
    #X, y = preprocessed_data  # Accessing the artifacts attribute of the StepArtifact object
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = model_trainer(X_train, y_train)
    #evaluation = model_evaluator(model, X_test, y_test)
    #accuracy, report = evaluation
    accuracy, report = model_evaluator(model, X_test, y_test)
    
    print(f'Accuracy: {accuracy}')
    print(report)
    
