# ML Pipeline Project

This project demonstrates a complete ML pipeline using ZenML for the Breast Cancer Wisconsin Diagnostic Dataset.

## Setup

1. Build the Docker image:

    ```bash
    docker build -t ml_pipeline_project .
    ```

2. Run the container:

    ```bash
    docker run -it --rm ml_pipeline_project
    ```

## Project Structure

- `data/`: Data loading scripts
- `features/`: Feature engineering scripts
- `models/`: Model training and evaluation scripts
- `pipelines/`: Pipeline configuration
- `main.py`: Main entry point for running the pipeline
- `Dockerfile`: Docker configuration
- `requirements.txt`: Python dependencies
- `setup.py`: Python package configuration

## Running the Pipeline

To run the pipeline, execute:

```bash
python main.py
