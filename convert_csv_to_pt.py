import pandas as pd
import torch
import os

def convert_csv_to_pt(directory):
    # Iterate over all CSV files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # Read the CSV file
            csv_path = os.path.join(directory, filename)
            df = pd.read_csv(csv_path)

            # Convert DataFrame to Tensor
            # Assuming all data is numerical. Modify as needed for other data types.
            tensor = torch.tensor(df.values)

            # Save as .pt file
            pt_path = os.path.join(directory, filename.replace('.csv', '.pt'))
            torch.save(tensor, pt_path)
            print(f"Converted {csv_path} to {pt_path}")


# Replace 'your_directory_path' with the path to your directory containing CSV files
convert_csv_to_pt('/mnt/nvme0n1/ICCV/ds-mil/datasets/cam-16/')
