import sys
import os

# Add the root directory of the project to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipelines.training_pipeline import training_pipeline
from pipelines.training_pipeline import (
    data_loader,
    data_preprocessor,
    model_trainer,
    model_evaluator,
)

pipeline = training_pipeline(
    data_loader=data_loader,
    data_preprocessor=data_preprocessor,
    model_trainer=model_trainer,
    model_evaluator=model_evaluator,
)

pipeline.run()
