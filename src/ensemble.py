import numpy as np
import pandas as pd
from kinoa import kinoa


id_col = 'encounter_id'
target_col = 'hospital_death'

files = [
    '__kinoa__/2020-01-17_16-28-11_LGB_0/submission0.csv',
    '__kinoa__/2020-01-17_17-36-23_LGB_0/submission0.csv',
    '__kinoa__/2020-01-18_00-47-58_LGB_0/submission0.csv',
    '__kinoa__/2020-01-18_02-04-52_LGB_0/submission0.csv',
    '__kinoa__/2020-01-20_14-15-06_Exp0/submission0.csv',
    '__kinoa__/2020-01-20_16-51-03_Exp0/submission0.csv',
    '__kinoa__/2020-01-20_17-05-59_Exp0/submission0.csv',
    '__kinoa__/2020-01-21_17-12-43_Exp0/submission0.csv',
    '__kinoa__/2020-02-14_17-47-04_Exp0/submission0.csv',
    '__kinoa__/2020-02-14_18-13-09_Exp0/submission0.csv',
    '__kinoa__/2020-02-14_18-38-24_Exp0/submission0.csv',
    '__kinoa__/2020-02-17_13-50-06_Exp0/submission0.csv',
    '__kinoa__/2020-02-17_15-10-51_Exp0/submission0.csv',
    '__kinoa__/2020-02-17_16-17-43_Exp0/submission0.csv',
    '__kinoa__/2020-02-17_17-58-07_Exp0/submission0.csv',
    '__kinoa__/2020-02-17_18-44-11_Exp0/submission0.csv',
    '__kinoa__/2020-02-17_23-53-17_Exp0/submission0.csv',
    '__kinoa__/2020-02-18_16-48-02_Exp0/submission0.csv',
    '__kinoa__/2020-02-18_17-08-18_Exp0/submission0.csv',
    '__kinoa__/2020-02-18_18-07-00_Exp0/submission0.csv',
    '__kinoa__/2020-02-18_18-33-20_Exp0/submission0.csv',
    '__kinoa__/2020-02-19_17-24-47_Exp1/submission1.csv',
    '__kinoa__/2020-02-19_18-34-47_Exp1/submission1.csv',
    '__kinoa__/2020-02-19_14-48-19_Exp0/submission0.csv',
    '__kinoa__/2020-02-19_18-13-35_Exp0/submission0.csv',
    '__kinoa__/2020-02-20_16-41-45_Exp0/submission0.csv'
]

cnt = len(files)

p_buf = []
for f in files:
    print(f)
    df = pd.read_csv(f)
    df.sort_values(id_col)
    p = df[target_col].values
    id_test = df[id_col].values
    p_buf.append(p)

test_preds = np.mean(p_buf, axis=0)

submission = pd.DataFrame()
submission[id_col] = id_test
submission[target_col] = test_preds
submission.to_csv('submission_ens.csv', index=False)

# Save backup
files = [
    'ensemble.py', 
    'ensemble.log',
    'submission_ens.csv'
]

experiment_name = 'ENS'
params = {}
params['n_models'] = cnt
scores = {}
scores['auc_mean'] = np.nan
scores['auc_std'] = np.nan
scores['kaggle'] = np.nan
other = {}
other['n_features'] = np.nan
other['n_splits'] = np.nan
comments = ''
kinoa.save(
    files,
    experiment_name=experiment_name,
    params=params,
    scores=scores,
    other=other,
    comments=comments,
    working_dir='',
    sort_log_by='experiment_datetime', 
    sort_log_ascending=True,
    columns_order={'scores.kaggle': -1, 'scores.auc_std': -2, 'scores.auc_mean': -3}
)
print('Done!')
