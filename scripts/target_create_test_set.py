import os
import sys

import numpy as np
import pandas as pd
import itertools

# Read CSV
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(parent_dir, "data","neck_test","neck_test_cmd_1m_full.csv")
df = pd.read_csv(csv_path)

# Randomly Select Rows
crop_df = df.sample(n=50, random_state=1)

# Save CSV
filepath = os.path.join(parent_dir, "data","neck_test","neck_test_cmd_1m_50.csv")
df.to_csv(filepath, index=False)
print('Saved to:', filepath)