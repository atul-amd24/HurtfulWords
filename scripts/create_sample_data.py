import os
import pandas as pd
import numpy as np
import glob
import argparse
import shutil


def random_patient_selection(input_file, output_file, n_patients=3500, random_seed=42):
    """
    Randomly select n_patients from MIMIC-III patients.csv and save to a new CSV file.
    
    Parameters:
    -----------
    input_file : str
        Path to the original MIMIC-III patients.csv file
    output_file : str
        Path where the output file with selected patients will be saved
    n_patients : int
        Number of patients to randomly select (default: 3500)
    random_seed : int
        Random seed for reproducibility (default: 42)
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Loading patients data from {input_file}...")
    
    # Read the original patients.csv file
    df_patients = pd.read_csv(input_file)
    
    # Get total number of patients
    total_patients = len(df_patients)
    print(f"Total number of patients in the dataset: {total_patients}")
    
    if total_patients < n_patients:
        raise ValueError(f"Requested {n_patients} patients but dataset only contains {total_patients}")
    
    # Randomly select n_patients
    print(f"Randomly selecting {n_patients} patients...")
    selected_indices = np.random.choice(total_patients, size=n_patients, replace=False)
    selected_patients = df_patients.iloc[selected_indices].copy()
    
    # Save selected patients to the output file
    print(f"Saving selected patients to {output_file}...")
    selected_patients.to_csv(output_file, index=False)
    
    print(f"Successfully saved {n_patients} randomly selected patients to {output_file}")
    
    return selected_patients

def filter_patient_data(input_dir, output_dir, patient_file, patient_id_column, chunk_size=10000):
    """
    Filter CSV files based on patient IDs with optimized memory usage.
    
    Args:
        input_dir: Directory containing CSV files
        output_dir: Directory where filtered files will be saved
        patient_file: Name of the file containing patient IDs
        patient_id_column: Column name for patient IDs
        chunk_size: Number of rows to process at once
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Full path to the patients file
    patients_file_path = os.path.join(input_dir, patient_file)
    
    # Read only the patient ID column to save memory
    print(f"Loading patient IDs from {patients_file_path}...")
    patient_ids = set(pd.read_csv(patients_file_path, usecols=[patient_id_column])[patient_id_column])
    print(f"Loaded {len(patient_ids)} unique patient IDs")

    # Find all CSV files in input directory (excluding the patients file)
    all_csv_files = [f for f in glob.glob(os.path.join(input_dir, "*.csv")) 
                    if os.path.basename(f) != patient_file]
    
    print(f"Found {len(all_csv_files)} additional CSV files to process")

    # Process each CSV file
    for csv_file in all_csv_files:
        file_name = os.path.basename(csv_file)
        output_filename = os.path.join(output_dir, file_name)
        
        print(f"Processing {file_name}...")
        
        try:
            # First check if the file has the patient ID column
            # Read just the header row to check columns
            header_df = pd.read_csv(csv_file, nrows=0)
            
            if patient_id_column not in header_df.columns:
                print(f"  - {patient_id_column} column not found in {file_name}, copying file as is")
                # Copy the file as is
                shutil.copy2(csv_file, output_filename)
                print(f"  - Copied original file to {output_filename}")
                continue
            
            # If we reach here, we need to filter the file
            print(f"  - Filtering {file_name} by {patient_id_column} in chunks...")
            
            # Process in chunks to save memory
            total_rows = 0
            filtered_rows = 0
            
            # Create an empty file first
            with open(output_filename, 'w') as f:
                pass
            
            # Process the file in chunks
            for chunk_number, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunk_size)):
                total_rows += len(chunk)
                
                # Filter the chunk
                filtered_chunk = chunk[chunk[patient_id_column].isin(patient_ids)]
                filtered_rows += len(filtered_chunk)
                
                # Write to output file (append mode except for first chunk)
                write_header = (chunk_number == 0)
                filtered_chunk.to_csv(output_filename, mode='a', header=write_header, index=False)
                
                # Print progress for large files
                if (chunk_number + 1) % 10 == 0:
                    print(f"  - Processed {chunk_number + 1} chunks ({total_rows} rows)...")
            
            print(f"  - Original file had {total_rows} rows")
            print(f"  - Filtered file has {filtered_rows} rows")
            print(f"  - Saved to {output_filename}")
            
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

    print(f"\nProcessing complete! All files are saved in the '{output_dir}' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter CSV files based on patient IDs with optimized memory usage.')
    parser.add_argument('--input_dir', type=str, default='.',
                        help='Directory containing the patients.csv and other CSV files (default: current directory)')
    parser.add_argument('--output_dir', type=str, default='filtered_patient_data',
                        help='Directory where filtered files will be saved (default: filtered_patient_data)')
    parser.add_argument('--patient_file', type=str, default='PATIENTS.csv',
                        help='Name of the CSV file containing patient IDs (default: PATIENTS.csv)')
    parser.add_argument('--id_column', type=str, default='SUBJECT_ID',
                        help='Name of the column containing patient IDs (default: SUBJECT_ID)')
    parser.add_argument('--chunk_size', type=int, default=10000,
                        help='Number of rows to process at once (default: 10000)')
    parser.add_argument('--num_of_patients', type=int, default=3500,
                        help='Number of rows to process at once (default: 3500)')    
    
    args = parser.parse_args()


    input_file = os.path.join(args.input_dir, args.patient_file)
    output_file = os.path.join(args.output_dir, args.patient_file)
    random_patient_selection(input_file, output_file, n_patients=args.num_of_patients, random_seed=42)

    filter_patient_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        patient_file=output_file,
        patient_id_column=args.id_column,
        chunk_size=args.chunk_size
    )
