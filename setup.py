from setuptools import setup, find_packages

setup(
    name="ml_pipeline_project",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "zenml==0.40.0",
        "scikit-learn",
        "pandas",
        "numpy",
        "flask"
    ],
)
