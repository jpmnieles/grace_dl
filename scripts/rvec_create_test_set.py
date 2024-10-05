import os
import sys

import numpy as np
import pandas as pd
import itertools


list_lnp=list(range(-35,36,5))
list_lnt=list(range(-10,31,10))
list_unt=list(range(40,-11,-10))
list_ep=list(range(-14,15,2))
list_et=list(range(20,-31,-5))

test_lnp=list(range(-32,34,5))
test_lnt=list(range(-5,31,10))
test_unt=list(range(35,-11,-10))
test_ep=list(range(-13,14,2))
test_et=list(range(17,-29,-5))

combinations = np.array(list(itertools.product(test_lnp, test_lnt, test_unt, test_ep, test_et, )))
crop_combi = combinations[np.random.choice(combinations.shape[0], 1500, replace=False)]

df = pd.DataFrame({
    'cmd_theta_lower_neck_pan': crop_combi[:,0],
    'cmd_theta_lower_neck_tilt': crop_combi[:,1],
    'cmd_theta_upper_neck_tilt': crop_combi[:,2],
    'cmd_theta_left_eye_pan': crop_combi[:,3],
    'cmd_theta_right_eye_pan': crop_combi[:,3],
    'cmd_theta_eyes_tilt': crop_combi[:,4],
})

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
filepath = os.path.join(parent_dir, "241005_075m_test_grace_dataset.csv")
df.to_csv(filepath, index=False)
print('Saved to:', filepath)