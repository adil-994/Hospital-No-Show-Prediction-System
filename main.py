# main.py
from src.data_processing.clean_medical import clean_medical_data
from src.data_processing.clean_weather import clean_weather_data
from src.data_processing.merge_datasets import merge_data
from src.features.feature_engineering import engineer_features
from src.models.train_model import execute_pipeline
from src.models.evaluate_model import run_final_evaluation

def run_project_pipeline():
    """
    Orchestrates the entire machine learning workflow from raw data to evaluation.
    Each step is isolated in its own module for better organization.
    """
    print("Project Start: Beginning the end-to-end pipeline.")
    print("-" * 50)

    # Phase 1: Data Cleaning
    print("Phase 1: Cleaning medical and weather datasets.")
    clean_medical_data()
    clean_weather_data()

    # Phase 2: Data Integration
    print("\nPhase 2: Merging datasets based on appointment dates.")
    merge_data()

    # Phase 3: Feature Engineering
    print("\nPhase 3: Transforming raw data into model-ready features.")
    engineer_features()

    # Phase 4: Model Competition (Experimentation)
    print("\nPhase 4: Running experiments and comparing algorithms.")
    execute_pipeline()

    # Phase 5: Final Evaluation
    print("\nPhase 5: Generating final visuals and classification reports.")
    run_final_evaluation()

    print("-" * 50)
    print("Project Complete: The model is now ready for deployment.")

if __name__ == "__main__":
    run_project_pipeline()