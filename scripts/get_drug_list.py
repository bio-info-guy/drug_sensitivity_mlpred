import pandas as pd
import sys

def get_drug_list(data_file):
    dataset = pd.read_csv(data_file)
    # Assuming the last 4686 columns are drug sensitivity data
    # This number comes from the skl_train_model.py script
    drug_y_all = dataset.iloc[:, -4686:]
    drug_list = drug_y_all.columns.tolist()
    for drug in drug_list:
        print(drug)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python get_drug_list.py <data_file_path>")
        sys.exit(1)
    data_file = sys.argv[1]
    get_drug_list(data_file)
