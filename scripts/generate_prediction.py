import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle
import pandas as pd
from tqdm import tqdm

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# List of model paths
model_paths = [
    '/Users/aravind/Premnisha/MS/dlh/HurtfulWords/data/models/adv_clinical_BERT_1_epoch_512',
    '/Users/aravind/Premnisha/MS/dlh/HurtfulWords/data/models/baseline_clinical_BERT_1_epoch_128',
    '/Users/aravind/Premnisha/MS/dlh/HurtfulWords/data/models/baseline_clinical_BERT_1_epoch_512'
]

# List of data files
data_files = {
    'inhosp_mort': '/Users/aravind/Premnisha/MS/dlh/HurtfulWords/data/finetuning/inhosp_mort.pkl',
    'phenotype_all': '/Users/aravind/Premnisha/MS/dlh/HurtfulWords/data/finetuning/phenotype_all.pkl'
}

# Batch size for inference
batch_size = 8

for model_path in model_paths:
    print(f"\n🔍 Loading model from {model_path}")
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()

    for dataset_name, file_path in data_files.items():
        print(f"\n📂 Processing dataset: {dataset_name}")
        df = pd.read_pickle(file_path)
        # print(f"cols: {df.columns}")
        # print(f"seqs: {df['seqs'].head()}")
        if 'seqs' not in df.columns:
            raise ValueError(f"Missing 'seqs' column in {file_path}")

        test_data = [' '.join(map(str, seq)) for seq in df['seqs'].tolist()]

        preds = {}
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(test_data), batch_size), desc="Predicting"):
                batch = test_data[i:i+batch_size]
                # Pass the batch of text to tokenizer
                inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                outputs = model(**inputs)
                batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                for j, pred in enumerate(batch_preds):
                    preds[i + j] = int(pred)  # Cast to int to ensure JSON/pickle compatibility

        # Create the directory for saving predictions if it doesn't exist
        output_dir = os.path.join(model_path, dataset_name)
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

        # Now, set the correct output file path
        output_path = os.path.join(output_dir, "preds.pkl")

        # Save predictions
        with open(output_path, 'wb') as f:
            pickle.dump(preds, f)

        print(f"✅ Saved predictions for '{dataset_name}' to '{output_path}'")
