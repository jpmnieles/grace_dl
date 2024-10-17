import os
import sys

import numpy as np
import pandas as pd

date = "241010_best"
data_dir="final"
csv_fn="241010_075m_best_train_grace_dataset.csv"
csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","data", data_dir, csv_fn)
temp_df = pd.read_csv(csv_file)

df_50 = temp_df[:50].copy().reset_index(drop=True)
df_100 = temp_df[:100].copy().reset_index(drop=True)
df_250 = temp_df[:250].copy().reset_index(drop=True)
df_500 = temp_df[:500].copy().reset_index(drop=True)
df_750 = temp_df[:750].copy().reset_index(drop=True)
df_1000 = temp_df[:1000].copy().reset_index(drop=True)

# Saving 50
res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','data','thesis', f'{date}_075m_grace_dataset_50.csv')
df_50.to_csv(res_path, index=False)

# Saving 100
res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','data','thesis', f'{date}_075m_grace_dataset_100.csv')
df_100.to_csv(res_path, index=False)

# Saving 250
res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','data','thesis', f'{date}_075m_grace_dataset_250.csv')
df_250.to_csv(res_path, index=False)

# Saving 500
res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','data','thesis', f'{date}_075m_grace_dataset_500.csv')
df_500.to_csv(res_path, index=False)

# Saving 750
res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','data','thesis', f'{date}_075m_grace_dataset_750.csv')
df_750.to_csv(res_path, index=False)

# Saving 1000
res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','data','thesis', f'{date}_075m_grace_dataset_1000.csv')
df_1000.to_csv(res_path, index=False)
