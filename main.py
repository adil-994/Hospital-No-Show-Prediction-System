# main.py
from src.data_processing.clean_medical import clean_medical_data
from src.data_processing.clean_weather import clean_weather_data
from src.data_processing.merge_datasets import merge_data
from src.features.feature_engineering import engineer_features
from src.visualizations.eda_plots import run_full_eda
from src.models.train_model import execute_pipeline
from src.models.tuning import execute_tuning_pipeline
from src.models.evaluate_model import run_final_evaluation

def run_project_pipeline():
    """
    Orchestrates the entire machine learning workflow.
    Each phase represents a key step in the Data Science lifecycle.
    """
    print("\n" + "="*50)
    print("PROJECT START: END-TO-END NO-SHOW PREDICTION")
    print("="*50)

    # Phase 1: Data Cleaning
    print("\nPhase 1: Cleaning medical and weather datasets.")
    clean_medical_data()
    clean_weather_data()

    # Phase 2: Data Integration
    print("\nPhase 2: Merging datasets based on appointment dates.")
    merge_data()

    # Phase 3: Feature Engineering
    print("\nPhase 3: Transforming raw data into model-ready features.")
    engineer_features()

    # Phase 4: Exploratory Data Analysis (EDA)
    print("\nPhase 4: Analyzing data trends and class imbalance.")
    run_full_eda()

    # Phase 5: Model Competition (Experimentation)
    print("\nPhase 5: Comparing algorithms (RF, SVM, Logistic) and Weather Impact.")
    execute_pipeline()

    # Phase 6: Hyperparameter Tuning & Serialization
    print("\nPhase 6: Optimizing the champion model and saving for deployment.")
    execute_tuning_pipeline()

    # Phase 7: Final Evaluation
    print("\nPhase 7: Generating final performance reports and visuals.")
    run_final_evaluation()

    print("\n" + "="*50)
    print("PROJECT COMPLETE: Tuned model is ready in 'artifacts/models/'")
    print("="*50)

if __name__ == "__main__":
    run_project_pipeline()