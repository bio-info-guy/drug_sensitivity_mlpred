import pandas as pd
import sys

def get_drug_list(data_file):
    dataset = pd.read_csv(data_file)
    # Dynamically determine drug columns based on pattern in column names
    all_columns = dataset.columns.tolist()
    drug_columns = [col for col in all_columns if col.startswith('BRD-')]
    
    # Use the identified drug columns to select data
    drug_y_all = dataset[drug_columns]
    drug_list = drug_y_all.columns.tolist()
    for drug in drug_list:
        print(drug)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python get_drug_list.py <data_file_path>")
        sys.exit(1)
    data_file = sys.argv[1]
    get_drug_list(data_file)
