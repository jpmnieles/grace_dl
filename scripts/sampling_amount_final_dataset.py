import os
import sys

import numpy as np
import pandas as pd

data_dir="final"
csv_fn="241003_075m_grace_dataset.csv"
csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","data", data_dir, csv_fn)
temp_df = pd.read_csv(csv_file)

df_50 = temp_df.sample(n=50, random_state=1).reset_index(drop=True)
df_100 = temp_df.sample(n=100, random_state=1).reset_index(drop=True)
df_500 = temp_df.sample(n=500, random_state=1).reset_index(drop=True)
df_1000 = temp_df.sample(n=1000, random_state=1).reset_index(drop=True)
df_5000 = temp_df.sample(n=5000, random_state=1).reset_index(drop=True)
df_10000 = temp_df.sample(n=10000, random_state=1).reset_index(drop=True)
df_30000 = temp_df.sample(n=30000, random_state=1).reset_index(drop=True)
df_50000 = temp_df.sample(n=50000, random_state=1).reset_index(drop=True)

# Saving 50
res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','data','thesis', '241005_075m_grace_dataset_50.csv')
df_50.to_csv(res_path, index=False)

# Saving 100
res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','data','thesis', '241005_075m_grace_dataset_100.csv')
df_100.to_csv(res_path, index=False)

# Saving 500
res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','data','thesis', '241005_075m_grace_dataset_500.csv')
df_500.to_csv(res_path, index=False)

# Saving 1000
res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','data','thesis', '241005_075m_grace_dataset_1000.csv')
df_1000.to_csv(res_path, index=False)

# Saving 5000
res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','data','thesis', '241005_075m_grace_dataset_5000.csv')
df_5000.to_csv(res_path, index=False)

# Saving 10000
res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','data','thesis', '241005_075m_grace_dataset_10000.csv')
df_10000.to_csv(res_path, index=False)

# Saving 30000
res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','data','thesis', '241005_075m_grace_dataset_30000.csv')
df_30000.to_csv(res_path, index=False)

# Saving 50000
res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','data','thesis', '241005_075m_grace_dataset_50000.csv')
df_50000.to_csv(res_path, index=False)