import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, recall_score
import matplotlib.pyplot as plt
from tabulate import tabulate

# Paths
PHENOTYPE_ALL_PATH = '/Users/aravind/Premnisha/MS/dlh/HurtfulWords/data/finetuning/sample/phenotype_all.pkl'
BASELINE_MODEL_DIR = '/Users/aravind/Premnisha/MS/dlh/HurtfulWords/data/models/baseline_clinical_BERT_1_epoch_512'
ADV_MODEL_DIR = '/Users/aravind/Premnisha/MS/dlh/HurtfulWords/data/models/adv_clinical_BERT_1_epoch_512'
torch.set_grad_enabled(False)

# Attribute list for bias evaluation
attribute_list = [
    # ("gender", "Male", "Female"),
    ("language_to_use", "English", "Other"),
    ("ethnicity_to_use", "White", "Other"),
    ("ethnicity_to_use", "Black", "Other"),
    ("ethnicity_to_use", "Hispanic", "Other"),
    ("ethnicity_to_use", "Asian", "Other"),
    ("insurance", "Medicare", "Other"),
    ("insurance", "Private", "Other"),
]

# Load dataset
phenotype_df = pd.read_pickle(PHENOTYPE_ALL_PATH)

class PhenotypeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label_column, group_value, max_length=512):
        self.texts = dataframe['seqs'].tolist()  # Assuming 'seqs' contains the text data
        self.labels = self.create_group_labels(dataframe[label_column], group_value)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def create_group_labels(self, column, group_value):
        # Create binary labels based on the group_value
        return [1 if val == group_value else 0 for val in column]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {key: val.squeeze(0) for key, val in encoding.items()}, torch.tensor(label)

# Define the function to get predictions
def get_predictions(model, tokenizer, dataframe, label_column, group_value):
    print(f"\n[DEBUG] Preparing predictions for label_column='{label_column}', group_value='{group_value}'")
    print(f"[DEBUG] Total number of rows before filtering: {len(dataframe)}")

    model.eval()
    dataset = PhenotypeDataset(dataframe, tokenizer, label_column, group_value)
    loader = DataLoader(dataset, batch_size=128)

    print(f"[DEBUG] Dataset created with {len(dataset)} samples")
    preds = []
    trues = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            inputs, labels = batch
            inputs = {k: v for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            batch_trues = labels.cpu().numpy()

            preds.extend(batch_preds)
            trues.extend(batch_trues)

            print(f"[DEBUG] Batch {batch_idx+1}: {len(batch_preds)} predictions")

    preds_array = np.array(preds)
    trues_array = np.array(trues)

    print(f"[DEBUG] Finished predictions: Total preds={len(preds_array)}, Total trues={len(trues_array)}")

    return preds_array, trues_array

# Load models and tokenizer
def load_model_and_tokenizer(model_dir):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

baseline_model, baseline_tokenizer = load_model_and_tokenizer(BASELINE_MODEL_DIR)
adv_model, adv_tokenizer = load_model_and_tokenizer(ADV_MODEL_DIR)

def specificity_score(y_true, y_pred):
    # Calculate True Negatives (TN) and False Positives (FP)
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    # Specificity: TN / (TN + FP)
    return TN / (TN + FP) if (TN + FP) != 0 else 0.0

def compute_fairness_metrics(preds, trues, label_column, group_value):
    accuracy = accuracy_score(trues, preds)
    recall = recall_score(trues, preds)
    specificity = specificity_score(trues, preds)
    
    print(f"Fairness Metrics for {label_column} ({group_value}):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    return accuracy, recall, specificity

# Evaluate performance group-wise
def evaluate_groupwise(preds, trues, df, attribute, group1, group2):
    group1_mask = df[attribute] == group1
    group2_mask = df[attribute] == group2

    tp1 = np.sum((preds[group1_mask] == 1) & (trues[group1_mask] == 1))
    tn1 = np.sum((preds[group1_mask] == 0) & (trues[group1_mask] == 0))
    fp1 = np.sum((preds[group1_mask] == 1) & (trues[group1_mask] == 0))
    fn1 = np.sum((preds[group1_mask] == 0) & (trues[group1_mask] == 1))
    n1 = group1_mask.sum()

    tp2 = np.sum((preds[group2_mask] == 1) & (trues[group2_mask] == 1))
    tn2 = np.sum((preds[group2_mask] == 0) & (trues[group2_mask] == 0))
    fp2 = np.sum((preds[group2_mask] == 1) & (trues[group2_mask] == 0))
    fn2 = np.sum((preds[group2_mask] == 0) & (trues[group2_mask] == 1))
    n2 = group2_mask.sum()

    parity_gap = (tp1 + fp1) / n1 - (tp2 + fp2) / n2
    recall_gap = (tp1 / (tp1 + fn1 + 1e-6)) - (tp2 / (tp2 + fn2 + 1e-6))
    specificity_gap = (tn1 / (tn1 + fp1 + 1e-6)) - (tn2 / (tn2 + fp2 + 1e-6))

    return {
        "parity_gap": parity_gap,
        "recall_gap": recall_gap,
        "specificity_gap": specificity_gap
    }

# Main evaluation
def full_evaluation():
    results = []

    # Loop through each attribute in the attribute list
    for label_column, group_value_1, group_value_2 in attribute_list:
        print(f"Evaluating fairness for {label_column} ({group_value_1} and {group_value_2})")

        for group_value in [group_value_1, group_value_2]:
            print(f"  -> Evaluating group: {group_value}")

            # Get baseline predictions
            baseline_preds, trues = get_predictions(baseline_model, baseline_tokenizer, phenotype_df, label_column, group_value)
            
            # Get adversarial model predictions
            adv_preds, _ = get_predictions(adv_model, adv_tokenizer, phenotype_df, label_column, group_value)

            print(f"    Start Computing fairness metrics for {group_value}")

            # Compute fairness metrics for both models
            baseline_metrics = compute_fairness_metrics(baseline_preds, trues, label_column, group_value)
            adv_metrics = compute_fairness_metrics(adv_preds, trues, label_column, group_value)

            # Store result
            results.append({
                'attribute': label_column,
                'group': group_value,
                'baseline_accuracy': baseline_metrics[0],
                'baseline_recall': baseline_metrics[1],
                'baseline_specificity': baseline_metrics[2],
                'adv_accuracy': adv_metrics[0],
                'adv_recall': adv_metrics[1],
                'adv_specificity': adv_metrics[2],
            })
            print("-------------RESULTS----------------")
            print(f"{results}")
            print("--------------------------")
    
    # Return results
    df_results = pd.DataFrame(results)
    print(tabulate(df_results, headers='keys', tablefmt='pretty', showindex=False))
    return df_results

# Visualization
def plot_gap_improvements(df_results):
    attributes = df_results['Attribute'] + " (" + df_results['Group1'] + " vs " + df_results['Group2'] + ")"
    x = np.arange(len(attributes))

    parity_improvement = df_results['Baseline Parity Gap'] - df_results['Adv Parity Gap']
    recall_improvement = df_results['Baseline Recall Gap'] - df_results['Adv Recall Gap']
    specificity_improvement = df_results['Baseline Specificity Gap'] - df_results['Adv Specificity Gap']

    width = 0.25
    plt.figure(figsize=(18, 7))
    plt.bar(x - width, parity_improvement, width, label="Parity Gap Improvement")
    plt.bar(x, recall_improvement, width, label="Recall Gap Improvement")
    plt.bar(x + width, specificity_improvement, width, label="Specificity Gap Improvement")

    plt.xlabel('Attribute and Groups')
    plt.ylabel('Gap Improvement (Baseline - Adv)')
    plt.title('Fairness Gap Improvements after Adversarial Debiasing')
    plt.xticks(x, attributes, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Run everything
if __name__ == "__main__":
    df_results = full_evaluation()
    plot_gap_improvements(df_results)
