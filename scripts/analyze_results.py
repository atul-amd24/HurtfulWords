import os
import pickle
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import pandas as pd

output_csv_path = "/Users/aravind/Premnisha/MS/dlh/HurtfulWords/data/evaluation_results.csv"
formatted_output_path = "/Users/aravind/Premnisha/MS/dlh/HurtfulWords/data/formatted_evaluation_results.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
MODEL_ROOT_DIR = '/Users/aravind/Premnisha/MS/dlh/HurtfulWords/data/models'
INHOSP_MORT_PATH = '/Users/aravind/Premnisha/MS/dlh/HurtfulWords/data/finetuning/sample/inhosp_mort.pkl'
PHENOTYPE_ALL_PATH = '/Users/aravind/Premnisha/MS/dlh/HurtfulWords/data/finetuning/sample/phenotype_all.pkl'
PHENOTYPE_FIRST_PATH = '/Users/aravind/Premnisha/MS/dlh/HurtfulWords/data/finetuning/sample/phenotype_first.pkl'

# Load the test datasets
def load_data():
    with open(INHOSP_MORT_PATH, 'rb') as f:
        inhosp_mort = pickle.load(f)
    with open(PHENOTYPE_ALL_PATH, 'rb') as f:
        phenotype_all = pickle.load(f)
    with open(PHENOTYPE_FIRST_PATH, 'rb') as f:
        phenotype_first = pickle.load(f)
    return inhosp_mort, phenotype_all, phenotype_first

# Load model and tokenizer
def load_model(model_dir):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

# Evaluate model on a dataset
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, 
    average_precision_score, log_loss, accuracy_score, 
    confusion_matrix, f1_score
)

def calculate_metrics(y_true, y_pred_probs, threshold=0.5):
    metrics = {}

    if len(y_true) == 0:
        return metrics

    y_pred_labels = (y_pred_probs >= threshold).astype(int)

    if len(np.unique(y_true)) > 1:
        metrics['auroc'] = roc_auc_score(y_true, y_pred_probs)

    metrics['precision'] = precision_score(y_true, y_pred_labels)
    metrics['recall'] = recall_score(y_true, y_pred_labels)
    metrics['auprc'] = average_precision_score(y_true, y_pred_probs)
    metrics['log_loss'] = log_loss(y_true, y_pred_probs, labels=[0,1])
    metrics['acc'] = accuracy_score(y_true, y_pred_labels)
    metrics['f1'] = f1_score(y_true, y_pred_labels, average='macro')

    cm = confusion_matrix(y_true, y_pred_labels, labels=[0,1])
    metrics['TN'] = cm[0,0]
    metrics['FP'] = cm[0,1]
    metrics['FN'] = cm[1,0]
    metrics['TP'] = cm[1,1]

    metrics['class_true_count'] = (y_true == 1).sum()
    metrics['class_false_count'] = (y_true == 0).sum()
    metrics['specificity'] = float(cm[0,0]) / (cm[0,0] + cm[0,1]) if metrics['class_false_count'] > 0 else 0

    metrics['pred_true_count'] = (y_pred_labels == 1).sum()
    metrics['nsamples'] = len(y_true)
    metrics['pred_prevalence'] = metrics['pred_true_count'] / float(len(y_true))
    metrics['actual_prevalence'] = metrics['class_true_count'] / float(len(y_true))

    return metrics


def evaluate_model(model, tokenizer, dataset, target, threshold=0.5):
    test_data = dataset[dataset['fold'] == 'test']
    print("Available columns in the dataset:", test_data.columns)

    X_test = test_data['seqs'].apply(str).values
    y_true = np.array(test_data[target].values)

    y_pred_labels = []
    y_pred_probs = []

    batch_size = 32

    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_texts = X_test[i:i+batch_size]
            batch_texts_clean = [str(text).replace('##', '') for text in batch_texts]

            encoded = tokenizer(batch_texts_clean, padding=True, truncation=True, return_tensors='pt', max_length=512)
            input_ids = encoded['input_ids'].to(DEVICE)
            attention_mask = encoded['attention_mask'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_pred_labels.extend(preds.cpu().numpy())
            y_pred_probs.extend(probs[:,1].cpu().numpy() if probs.shape[1] == 2 else probs.cpu().numpy())

    y_pred_probs = np.array(y_pred_probs)
    y_pred_labels = np.array(y_pred_labels)

    metrics = calculate_metrics(y_true, y_pred_probs, threshold=threshold)

    return metrics

def format_results(results_df):
    # Define model name mapping for columns
    model_mapping = {
        'baseline_clinical_BERT_1_epoch_128': 'Baseline 128',
        'baseline_clinical_BERT_1_epoch_512': 'Baseline 512',
        'adv_clinical_BERT_1_epoch_512': 'Adv BERT 512'
    }
    
    # Create a list to store the formatted results
    formatted_results = []
    
    # Get unique dataset-target combinations
    dataset_targets = results_df[['dataset_name', 'target_name']].drop_duplicates().values
    
    # For each dataset-target combination
    for dataset_name, target_name in dataset_targets:
        # Filter data for this dataset-target
        dataset_df = results_df[(results_df['dataset_name'] == dataset_name) & 
                               (results_df['target_name'] == target_name)]
        
        # Get list of all metrics
        metrics = dataset_df['metric_name'].unique()
        
        # For each metric
        for metric in metrics:
            row = {'Metric': metric}
            
            # For each model, get the value for this metric
            for model_name in model_mapping.keys():
                value = dataset_df[(dataset_df['model_name'] == model_name) & 
                                  (dataset_df['metric_name'] == metric)]['metric_value'].values
                
                if len(value) > 0:
                    row[model_mapping[model_name]] = value[0]
                else:
                    row[model_mapping[model_name]] = None
            
            # Add dataset and target information
            row['Dataset'] = dataset_name
            row['Target'] = target_name
            
            formatted_results.append(row)
    
    # Convert to DataFrame
    formatted_df = pd.DataFrame(formatted_results)
    
    # Reorder columns
    col_order = ['Dataset', 'Target', 'Metric'] + [model_mapping[m] for m in model_mapping.keys()]
    formatted_df = formatted_df[col_order]
    
    return formatted_df

def main():
    inhosp_mort, phenotype_all, phenotype_first = load_data()

    model_folders = [
        'baseline_clinical_BERT_1_epoch_128',
        'baseline_clinical_BERT_1_epoch_512',
        'adv_clinical_BERT_1_epoch_512'
    ]

    test_dataset_target = {
        "inhosp_mort": (inhosp_mort, "inhosp_mort"),
        "pheno_all": (phenotype_all, "any_acute"),
        "pheno_first": (phenotype_first, "any_acute")
    }

    results = []

    for model_folder in model_folders:
        model_path = os.path.join(MODEL_ROOT_DIR, model_folder)
        print(f"\n=== Evaluating model: {model_folder} ===")

        model, tokenizer = load_model(model_path)
        for dataset_name, (dataset, target_name) in test_dataset_target.items():
            print(f"\nEvaluating on {dataset_name} Dataset")
            metrics = evaluate_model(model, tokenizer, dataset, target_name)

            for metric_name, metric_value in metrics.items():
                results.append({
                    "model_name": model_folder,
                    "dataset_name": dataset_name,
                    "target_name": target_name,
                    "metric_name": metric_name,
                    "metric_value": round(metric_value, 4)
                })


    df = pd.DataFrame(results)
    # df.to_csv(output_csv_path, index=False)

    formatted_df = format_results(df)
    formatted_df.to_csv(formatted_output_path, index=False)

    print(f"\n✅ Evaluation results saved to {formatted_output_path}")

if __name__ == "__main__":
    main()


