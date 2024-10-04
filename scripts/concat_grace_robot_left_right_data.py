import os
import sys

import glob
import numpy as np
import pandas as pd

def preprocess_csv(data_dir, depth, eye):
    # Read CSV
    files_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",'data', data_dir, f"*{depth}*", f"*{eye}*.csv")
    csv_files = glob.glob(files_path)
    
    dfs = []
    for csv_file in csv_files:
        temp_df = pd.read_csv(csv_file)
        dfs.append(temp_df)
    df = pd.concat(dfs, ignore_index=True)

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','data','structured',f"{depth}m_{eye}_concat.csv")
    df.to_csv(data_path , index=False)
    print('Saved to:',data_path)

if __name__ == '__main__':
    preprocess_csv('raw','d1','left')
    preprocess_csv('raw','d1','right')