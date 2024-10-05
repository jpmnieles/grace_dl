import os
import sys

import glob
import numpy as np
import pandas as pd

def preprocess_csv(data_dir, depth):
    # Read CSV
    eye = 'left'
    files_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",'data', data_dir, f"*{depth}*", f"*{eye}*.csv")
    csv_files = glob.glob(files_path)
    
    left_dfs = []
    right_dfs = []
    for csv_file in csv_files:
        temp_df1 = pd.read_csv(csv_file)
        left_dfs.append(temp_df1)

        right_csv = csv_file[:78] + 'right' + csv_file[82:]
        temp_df2 = pd.read_csv(right_csv)
        right_dfs.append(temp_df2)
        
    left_df = pd.concat(left_dfs, ignore_index=True)
    right_df = pd.concat(right_dfs, ignore_index=True)

    eye = 'left'
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','data','structured',f"test_{depth}m_{eye}_concat.csv")
    left_df.to_csv(data_path, index=False)
    print('Saved to:',data_path)

    eye = 'right'
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','data','structured',f"test_{depth}m_{eye}_concat.csv")
    right_df.to_csv(data_path, index=False)
    print('Saved to:',data_path)

if __name__ == '__main__':
    preprocess_csv('test','d075')