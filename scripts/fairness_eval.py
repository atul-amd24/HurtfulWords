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
    ("gender", "Male", "Female"),
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
def evaluate_groupwise(df):
    results = []

    for attribute, group1, group2 in attribute_list:
        attr_df = df[df['attribute'] == attribute]

        g1 = attr_df[attr_df['group'] == group1]
        g2 = attr_df[attr_df['group'] == group2]

        if g1.empty or g2.empty:
            print(f"[WARN] Missing data for {attribute} comparison: {group1} vs {group2}")
            continue

        g1 = g1.iloc[0]
        g2 = g2.iloc[0]

        result = {
            "attribute": attribute,
            "group1": group1,
            "group2": group2,
            "baseline_parity_gap": g1['baseline_accuracy'] - g2['baseline_accuracy'],
            "baseline_recall_gap": g1['baseline_recall'] - g2['baseline_recall'],
            "baseline_specificity_gap": g1['baseline_specificity'] - g2['baseline_specificity'],
            "adv_parity_gap": g1['adv_accuracy'] - g2['adv_accuracy'],
            "adv_recall_gap": g1['adv_recall'] - g2['adv_recall'],
            "adv_specificity_gap": g1['adv_specificity'] - g2['adv_specificity'],
        }

        print("-------------RESULTS----------------")
        print(f"{results}")
        print("--------------------------")

        results.append(result)

    return pd.DataFrame(results)

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

# # Visualization
def plot_gap_improvements(df_results):
    attributes = df_results['attribute'] + " (" + df_results['group1'] + " vs " + df_results['group2'] + ")"
    x = np.arange(len(attributes))

    parity_improvement = df_results['baseline_parity_gap'] - df_results['adv_parity_gap']
    recall_improvement = df_results['baseline_recall_gap'] - df_results['adv_recall_gap']
    specificity_improvement = df_results['baseline_specificity_gap'] - df_results['adv_specificity_gap']

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

if __name__ == "__main__":
    # df_results = full_evaluation()
    df_results=[{'attribute': 'gender', 'group': 'Male', 'baseline_accuracy': 0.0, 'baseline_recall': 0.0, 'baseline_specificity': 0.0, 'adv_accuracy': 0.9942418426103646, 'adv_recall': 0.0, 'adv_specificity': 0.9942418426103646}, {'attribute': 'gender', 'group': 'Female', 'baseline_accuracy': 0.0, 'baseline_recall': 0.0, 'baseline_specificity': 0.0, 'adv_accuracy': 0.9942418426103646, 'adv_recall': 0.0, 'adv_specificity': 0.9942418426103646},{'attribute': 'language_to_use', 'group': 'English', 'baseline_accuracy': 0.6084452975047985, 'baseline_recall': 0.3907563025210084, 'baseline_specificity': 0.7915194346289752, 'adv_accuracy': 0.5143953934740882, 'adv_recall': 0.21428571428571427, 'adv_specificity': 0.7667844522968198}, {'attribute': 'language_to_use', 'group': 'Other', 'baseline_accuracy': 0.6775431861804223, 'baseline_recall': 0.3181818181818182, 'baseline_specificity': 0.710691823899371, 'adv_accuracy': 0.7332053742802304, 'adv_recall': 0.25, 'adv_specificity': 0.7777777777777778}, {'attribute': 'ethnicity_to_use', 'group': 'White', 'baseline_accuracy': 0.708253358925144, 'baseline_recall': 0.0, 'baseline_specificity': 0.708253358925144, 'adv_accuracy': 0.7754318618042226, 'adv_recall': 0.0, 'adv_specificity': 0.7754318618042226}, {'attribute': 'ethnicity_to_use', 'group': 'Other', 'baseline_accuracy': 0.708253358925144, 'baseline_recall': 0.0, 'baseline_specificity': 0.708253358925144, 'adv_accuracy': 0.7754318618042226, 'adv_recall': 0.0, 'adv_specificity': 0.7754318618042226}, {'attribute': 'ethnicity_to_use', 'group': 'Black', 'baseline_accuracy': 0.708253358925144, 'baseline_recall': 0.0, 'baseline_specificity': 0.708253358925144, 'adv_accuracy': 0.7754318618042226, 'adv_recall': 0.0, 'adv_specificity': 0.7754318618042226}, {'attribute': 'ethnicity_to_use', 'group': 'Other', 'baseline_accuracy': 0.708253358925144, 'baseline_recall': 0.0, 'baseline_specificity': 0.708253358925144, 'adv_accuracy': 0.7754318618042226, 'adv_recall': 0.0, 'adv_specificity': 0.7754318618042226}, {'attribute': 'ethnicity_to_use', 'group': 'Hispanic', 'baseline_accuracy': 0.708253358925144, 'baseline_recall': 0.0, 'baseline_specificity': 0.708253358925144, 'adv_accuracy': 0.7754318618042226, 'adv_recall': 0.0, 'adv_specificity': 0.7754318618042226}, {'attribute': 'ethnicity_to_use', 'group': 'Other', 'baseline_accuracy': 0.708253358925144, 'baseline_recall': 0.0, 'baseline_specificity': 0.708253358925144, 'adv_accuracy': 0.7754318618042226, 'adv_recall': 0.0, 'adv_specificity': 0.7754318618042226}, {'attribute': 'ethnicity_to_use', 'group': 'Asian', 'baseline_accuracy': 0.708253358925144, 'baseline_recall': 0.0, 'baseline_specificity': 0.708253358925144, 'adv_accuracy': 0.7754318618042226, 'adv_recall': 0.0, 'adv_specificity': 0.7754318618042226}, {'attribute': 'ethnicity_to_use', 'group': 'Other', 'baseline_accuracy': 0.708253358925144, 'baseline_recall': 0.0, 'baseline_specificity': 0.708253358925144, 'adv_accuracy': 0.7754318618042226, 'adv_recall': 0.0, 'adv_specificity': 0.7754318618042226}, {'attribute': 'insurance', 'group': 'Medicare', 'baseline_accuracy': 0.5547024952015355, 'baseline_recall': 0.3412698412698413, 'baseline_specificity': 0.7546468401486989, 'adv_accuracy': 0.5028790786948176, 'adv_recall': 0.21825396825396826, 'adv_specificity': 0.7695167286245354}, {'attribute': 'insurance', 'group': 'Other', 'baseline_accuracy': 0.708253358925144, 'baseline_recall': 0.0, 'baseline_specificity': 0.708253358925144, 'adv_accuracy': 0.7754318618042226, 'adv_recall': 0.0, 'adv_specificity': 0.7754318618042226}, {'attribute': 'insurance', 'group': 'Private', 'baseline_accuracy': 0.510556621880998, 'baseline_recall': 0.22162162162162163, 'baseline_specificity': 0.6696428571428571, 'adv_accuracy': 0.581573896353167, 'adv_recall': 0.22702702702702704, 'adv_specificity': 0.7767857142857143}, {'attribute': 'insurance', 'group': 'Other', 'baseline_accuracy': 0.708253358925144, 'baseline_recall': 0.0, 'baseline_specificity': 0.708253358925144, 'adv_accuracy': 0.7754318618042226, 'adv_recall': 0.0, 'adv_specificity': 0.7754318618042226}]

    df_results = pd.DataFrame(df_results)
    output_path = "/Users/aravind/Premnisha/MS/dlh/HurtfulWords/data/fairness_results.csv"
    df_results.to_csv(output_path, index=False)
    print(f"[INFO] Fairness metrics results written to '{output_path}'")
    
    metric_cols = ['baseline_accuracy', 'baseline_recall', 'baseline_specificity',
                   'adv_accuracy', 'adv_recall', 'adv_specificity']
    
    df_results[metric_cols] = df_results[metric_cols].applymap(lambda x: round(x * 100, 2))

    df_comp = evaluate_groupwise(df_results)
    output_path_comp = "/Users/aravind/Premnisha/MS/dlh/HurtfulWords/data/compared_fairness_results.csv"
    df_comp.to_csv(output_path_comp, index=False)
    print(f"[INFO] Compared Fairness metrics results written to '{output_path}'")
    plot_gap_improvements(df_comp)