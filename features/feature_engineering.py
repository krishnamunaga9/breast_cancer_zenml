from typing import Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
    features = df.drop(columns=['target'])
    target = df['target']
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features, target
