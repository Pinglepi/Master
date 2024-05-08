import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    # Define the base directory where prediction files are stored
    base_dir = 'Exports/Pred_values'
    base_dir_2 = 'Exports'

    # List all species directories
    dataset_list = ["General", "Bacillales", "Corynebacteriales", "Burkholderiales", "Lactobacillales", "Enterobacterales", "Pfam_1", "Pfam_2", "Pfam_3", "Pfam_4", "Pfam_5", 'Pfam_1-20']

    # Prepare a list to hold all DataFrames
    all_data_predictions = []
    all_data_performance = []

    # Iterate over each species directory for prediction files
    for dataset in dataset_list:
        pred_species_dir = os.path.join(base_dir, dataset)
        
        if os.path.exists(pred_species_dir):
            for file_name in os.listdir(pred_species_dir):
                if file_name.endswith('_predictions.csv'):
                    file_path = os.path.join(pred_species_dir, file_name)
                    df = pd.read_csv(file_path)
                    model_name = file_name.replace('_predictions.csv', '')
                    df['dataset'] = dataset
                    df['Model'] = model_name
                    all_data_predictions.append(df)

    # Process performance files, assuming they are located directly in the 'Exports' directory
    perf_species_dir = base_dir_2
    if os.path.exists(perf_species_dir):
        for file_name in os.listdir(perf_species_dir):
            if file_name.endswith('_results.csv'):
                file_path = os.path.join(perf_species_dir, file_name)
                df = pd.read_csv(file_path)
                model_name = file_name.replace('_results.csv', '')
                # Assuming performance data does not need to be tagged with 'Species'
                df['Dataset'] = model_name
                all_data_performance.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df_predictions = pd.concat(all_data_predictions, ignore_index=True)
    combined_df_performance = pd.concat(all_data_performance, ignore_index=True)

    return combined_df_predictions, combined_df_performance

def plot_roc_curve(df_predictions, model_name, dataset):

    sns.set(font_scale=2)

    # Filter the DataFrame for the specified model and dataset
    model_df = df_predictions[(df_predictions['Model'] == model_name) & (df_predictions['dataset'] == dataset)]

    # True labels and predicted probabilities
    true_labels = model_df['True_Labels']
    predicted_probabilities = model_df['Predicted_Probabilities']

    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probabilities)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name} on {dataset}')
    plt.legend(loc="lower right")

    # Save the plot before showing it in the folder Exports/images under the name of the model and dataset
    plt.savefig(f'Exports/Images/Roc/{model_name}_{dataset}_roc_curve.png')

    return None

def plot_confustion_matrix(df_predictions, model_name, dataset):
    
    sns.set(font_scale=2)

    # Filter the DataFrame for the specified model and dataset
    model_df = df_predictions[(df_predictions['Model'] == model_name) & (df_predictions['dataset'] == dataset)]

    # True labels and predicted probabilities
    true_labels = model_df['True_Labels']
    predicted_labels = model_df['Predicted_Labels']

    cm = confusion_matrix(true_labels, predicted_labels)

    # Plot confusion matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix for {model_name} on {dataset}')
    
    # Save the plot before showing it in the folder Exports/images under the name of the model and dataset
    plt.savefig(f'Exports/Images/Matrix/{model_name}_{dataset}_confusion_matrix.png')

    return None

def plot_distribution_probability(df_predictions, model_name, dataset):
    
    sns.set(font_scale=2)

    model_df = df_predictions[(df_predictions['Model'] == model_name) & (df_predictions['dataset'] == dataset)]
    # remake them with 10 bins instead
    bins = np.linspace(0, 1, 11)
    labels = [f'{i}-{i+10}%' for i in range(0, 100, 10)]

    # make the labels the middle instead of range
    labels = [f'{i+5}%' for i in range(0, 100, 10)]

    # Assign bins to predicted probabilities
    model_df['Bin'] = pd.cut(model_df['Predicted_Probabilities'], bins=bins, labels=labels, include_lowest=True)

    # Calculate accuracy for each bin
    accuracy_per_bin = model_df.groupby('Bin').apply(
        lambda x: np.mean(x['True_Labels'] == x['Predicted_Labels'])
    )

    # Count for each bin
    count_per_bin = model_df['Bin'].value_counts().sort_index()

    # Plotting the results
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10))

    # Accuracy plot
    axes[0].bar(accuracy_per_bin.index.astype(str), accuracy_per_bin.values, color='skyblue')
    axes[0].set_title(f'Accuracy for {model_name} on {dataset}')
    axes[0].set_xlabel('Predicted Probability Bins')
    axes[0].set_ylabel('Accuracy')
    # limit the y-axis to 0.5 to 1
    axes[0].set_ylim(0.5, 1)

    # Count plot
    axes[1].bar(count_per_bin.index.astype(str), count_per_bin.values, color='lightgreen')
    axes[1].set_title(f'Count for {model_name} on {dataset}')
    axes[1].set_xlabel('Predicted Probability Bins')
    axes[1].set_ylabel('Count')


    plt.tight_layout()
    # save the plot
    plt.savefig(f'Exports/Images/Dist/{model_name}_{dataset}_distribution_probability.png')
    return None

if __name__ == '__main__':
    # Load data
    df_predictions, df_performance = load_data()

    models = ['CNN_super', 'TM', 'XGBoost', 'DNN', 'TM_conv', 'CNN_simple']
    datasets = ['Corynebacteriales', 'Lactobacillales', 'General', 'Pfam_1', 'Pfam_1-20']

    for dataset in datasets:
        for model in models:
            plot_roc_curve(df_predictions, model, dataset)
            plot_confustion_matrix(df_predictions, model, dataset)
            plot_distribution_probability(df_predictions, model, dataset)
9