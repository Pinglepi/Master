import logging
import os
logging.basicConfig(level=logging.CRITICAL)

import numpy as np
import pandas as pd

from One_hot_encoding import one_hot_encode_sequences_in_batches
from sklearn.model_selection import train_test_split



from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time
import argparse
from TF_models import *

# Define function to save results to CSV
def save_results_to_csv(species, species_results):
    result_df = pd.DataFrame(species_results)
    result_df.to_csv(f"Exports/{species}_results.csv", index=False)

# Main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a dataset")
    parser.add_argument("species", type=str, help="Dataset to process")
    args = parser.parse_args()
    species = args.species
    print(f"Processing {species}...")

    pred_values_dir = f"Exports/Pred_values/{species}"
    os.makedirs(pred_values_dir, exist_ok=True)

    data = pd.read_csv(f"Data/{species}_lorfs.csv")
    data = data[(data["length"] >= 100) & (data["length"] <= 200)]
    data_pos_full = data[data["has_hmm"] == 1]
    data_neg_full = data[data["has_hmm"] == 0]

    # Check the number of positives and adjust if necessary
    data_length = 250000 # change this when adjusting the number of samples
    if len(data_pos_full) >= data_length:
        data_pos = data_pos_full.sample(n=data_length, random_state=1)
    else:
        data_pos = data_pos_full

    # Balance the negative data based on the number of positives
    data_neg = data_neg_full.sample(n=min(len(data_neg_full), len(data_pos)), random_state=1)

    # Ensure that the number of positive samples matches the number of negative samples, if necessary
    if len(data_neg) < len(data_pos):
        data_pos = data_pos.sample(n=len(data_neg), random_state=1)
    data = pd.concat([data_pos, data_neg])
    data = data.sample(frac=1, random_state=1).reset_index(drop=True)

    X, y = one_hot_encode_sequences_in_batches(data, batch_size=64, padding=200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.2, random_state=42, stratify=y_test)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    models_results = []
    models = { "TM": Create_tsetlin, "TM_conv": Create_tsetlin_convolution, "DNN": Create_DNN, "XGBoost": Create_XGBoost, "CNN_simple": Create_CNN_simple, "CNN_super": Create_CNN_super
}  # Add more models here
    #  , 
    for model_name, model_fn in models.items():
        train_start = time.time()
        model = model_fn(X_train, y_train, X_val, y_val)
        train_end = time.time()

        if "XGBoost" in model_name:
            dtest = xgb.DMatrix(X_test_flat, label=y_test)
            test_start = time.time()
            y_pred_probs = model.predict(dtest)
            y_pred = (y_pred_probs > 0.5).astype(int)
            test_end = time.time()
        else:
            test_start = time.time()
            y_pred_probs = model.predict(X_test_flat if "DNN" in model_name or "TM" in model_name else X_test)
            y_pred = (y_pred_probs > 0.5).astype(int)
            test_end = time.time()

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        training_time = train_end - train_start
        testing_time = test_end - test_start

        models_results.append({
            "Model": model_name, "Accuracy": acc, "F1": f1, "Precision": precision, "Recall": recall,
            "Training Time": training_time, "Testing Time": testing_time
        })

        # Save predictions and true values
        results_df = pd.DataFrame({
            'True_Labels': y_test,
            'Predicted_Labels': y_pred.flatten(),
            'Predicted_Probabilities': y_pred_probs.flatten()
        })
        results_filename = f"{pred_values_dir}/{model_name}_predictions.csv"
        results_df.to_csv(results_filename, index=False)

    save_results_to_csv(species, models_results)
