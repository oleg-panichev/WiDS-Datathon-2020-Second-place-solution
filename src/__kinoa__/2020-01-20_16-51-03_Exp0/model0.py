import datetime
import gc
import numpy as np
import os
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK']='True' # MacOS fix for libomp issues (https://github.com/dmlc/xgboost/issues/1715)

import lightgbm as lgb
import xgboost as xgb

from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold, RepeatedKFold, GroupKFold, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import NuSVC
from tqdm import tqdm as tqdm

from kinoa import kinoa
from scipy.stats import ttest_ind, ks_2samp
from sklearn.impute import SimpleImputer


def dprint(*args, **kwargs):
    print("[{}] ".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) + \
        " ".join(map(str,args)), **kwargs)


dprint('PID: {}'.format(os.getpid()))
script_id = 0

data_path = '../input/'

id_col = 'encounter_id'
target_col = 'hospital_death'
fillna_with_est = False
train_lgbm = True
train_xgb = False
# train_catboost = False

train = pd.read_csv(os.path.join(data_path, 'training_v2.csv'))
test = pd.read_csv(os.path.join(data_path, 'unlabeled.csv'))


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


# Drop constant columns
constant_cols = []
for c in train.columns:
    if train[c].nunique(dropna=False) < 2:
        constant_cols.append(c)
print(f'constant_cols in train: {constant_cols}')
train.drop(constant_cols, axis=1, inplace=True)
test.drop(constant_cols, axis=1, inplace=True)

constant_cols = []
for c in test.columns:
    if c != target_col and test[c].nunique(dropna=False) < 2:
        constant_cols.append(c)
print(f'constant_cols in test: {constant_cols}')
train.drop(constant_cols, axis=1, inplace=True)
test.drop(constant_cols, axis=1, inplace=True)

# imputer = SimpleImputer(missing_values=np.nan, strategy='median', copy=False)
# imputer.fit(train.values)
# # output is in numpy, so convert to df
# train = pd.DataFrame(imp_mean.transform(train), columns=train.columns)
# test = pd.DataFrame(imp_mean.transform(test), columns=test.columns)


# Add estimated variables to the dataset
est_cols = [
    {
        'name': 'weight', 
        'fillna': False, 
    },
    {
        'name': 'height',
        'fillna': False, 
    },
    {
        'name': 'apache_4a_hospital_death_prob',
        'fillna': False, 
    },
    # {
    #     'name': 'apache_4a_icu_death_prob',
    #     'fillna': False, 
    # }, # Worse
    # {
    #     'name': 'urineoutput_apache',
    #     'fillna': False, 
    # }, # Worse
    # {
    #     'name': 'bmi',
    #     'fillna': False, 
    # }, # Worse
    # {
    #     'name': 'glucose_apache',
    #     'fillna': False, 
    # }, # Worse
]

for c in est_cols:
    df = pd.read_csv(f'{c["name"]}_est.csv')
    train = train.merge(df, on=id_col, how='left')
    test = test.merge(df, on=id_col, how='left')

    if c['fillna']:
        train.loc[train[c['name']].isnull(), c['name']] = train[c['name'] + '_est']
        test.loc[test[c['name']].isnull(), c['name']] = test[c['name'] + '_est']

        train.drop([c['name'] + '_est'], axis=1, inplace=True)
        test.drop([c['name'] + '_est'], axis=1, inplace=True)

dprint(train.shape, test.shape)

# Extract features
def extract_features(df):
    df['d1_temp_minmax'] = df['d1_temp_max'] - df['d1_temp_min']
    df['d1_glucose_minmax'] = df['d1_glucose_max'] - df['d1_glucose_min']
    df['d1_resprate_minmax'] = df['d1_resprate_max'] - df['d1_resprate_min']
    df['d1_spo2_minmax'] = df['d1_spo2_max'] - df['d1_spo2_min']
    df['d1_platelets_minmax'] = df['d1_platelets_max'] - df['d1_platelets_min']

    # df['d1_heartrate_minmax'] = df['d1_heartrate_max'] - df['d1_heartrate_min']
    # df['h1_heartrate_minmax'] = df['h1_heartrate_max'] - df['h1_heartrate_min']

    # df['h1_temp_minmax'] = df['h1_temp_max'] - df['h1_temp_min']
    # df['h1_glucose_minmax'] = df['h1_glucose_max'] - df['h1_glucose_min']
    # df['h1_resprate_minmax'] = df['h1_resprate_max'] - df['h1_resprate_min']
    # df['h1_spo2_minmax'] = df['h1_spo2_max'] - df['h1_spo2_min']
    # df['h1_platelets_minmax'] = df['h1_platelets_max'] - df['h1_platelets_min']

    # df['abmi'] = df['age']*100*100*df['weight']/df['height']/df['height']

    df['apache_4a_hospicu_death_prob'] = df['apache_4a_hospital_death_prob'] + df['apache_4a_icu_death_prob']
    # df['apache_4a_hospicu_death_prob_m'] = df['apache_4a_hospital_death_prob'] * df['apache_4a_icu_death_prob']
    df['age_group'] = df['age']//5
    df['weight_group'] = df['weight']//5

    # df['hr_a'] = df['d1_heartrate_max']/df['age']
    # df['hr_w'] = df['d1_heartrate_max']/df['weight']

    if fillna_with_est:
        df['bmi'] = 100*100*df['weight']/df['height']/df['height']
    else:
        df['bmi_w_est'] = 100*100*df['weight_est']/df['height']/df['height']
        df['bmi_h_est'] = 100*100*df['weight']/df['height_est']/df['height_est']
        df['bmi_wh_est'] = 100*100*df['weight_est']/df['height_est']/df['height_est']

    # df['agi'] = df['weight']/df['age']
    # df['hrw'] = df['d1_heartrate_max']/df['weight']

    # cols = ['temp_apache', 'd1_temp_max', 'd1_temp_min', 'h1_temp_max', 'h1_temp_min']
    # for c in cols:
    #     df[c] = df[c]/36.6
    # df['apache_3j_bodysystem_apache_2_bodysystem'] = \
    #     df.apply(lambda r: str(r['apache_3j_bodysystem']) + '_' + str(r['apache_2_bodysystem']), axis=1)
    pass

extract_features(train)
extract_features(test)   

train['is_test'] = 0
test['is_test'] = 1
df_all = pd.concat([train, test], axis=0)

dprint('Label Encoder...')
cols = [f_ for f_ in df_all.columns if df_all[f_].dtype == 'object']
print(cols)
cnt = 0
for c in tqdm(cols):
    if c != id_col:
        # print(c)
        le = LabelEncoder()
        df_all[c] = le.fit_transform(df_all[c].astype(str))
        cnt += 1

        del le
dprint('len(cols) = {}'.format(cnt))


gfs = ['hospital_id', 'icu_id', 'age_group', 'apache_3j_diagnosis', 'gender', 'ethnicity', 'apache_3j_bodysystem']#, 'apache_2_bodysystem'] #+ \
    # ['hospital_admit_source', 'icu_admit_source', 'icu_stay_type', 'icu_type', 'apache_2_bodysystem']
ffs = ['apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob', 'bmi', 'd1_heartrate_min']
# ffs = ['apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob', 'bmi', 'bmi_w_est', 'bmi_h_est', 'bmi_wh_est', 'weight', 'height']


for gf in gfs:
    for ff in ffs:
        g = df_all.groupby(gf)[ff].agg(['mean', 'std', 'min', 'max']).reset_index()
        g.rename({'mean': f'{gf}_{ff}__mean', 'std': f'{gf}_{ff}__std', 'min': f'{gf}_{ff}__min', 'max': f'{gf}_{ff}__max'}, axis=1, inplace=True)
        df_all = df_all.merge(g, on=gf, how='left')

train = df_all.loc[df_all['is_test'] == 0].drop(['is_test'], axis=1)
test = df_all.loc[df_all['is_test'] == 1].drop(['is_test'], axis=1)

del df_all
gc.collect()

# # Fill nans
# train.fillna(train.mean(), inplace=True)
# test.fillna(train.mean(), inplace=True)

features = list(train.columns.values)
features.remove(id_col)
features.remove(target_col)


# Build the model
cnt = 0
p_buf = []
n_splits = 4
n_repeats = 1
kf = RepeatedKFold(
    n_splits=n_splits, 
    n_repeats=n_repeats, 
    random_state=0)
err_buf = []   
undersampling = 0

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 8,
    'learning_rate': 0.05, 
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'lambda_l1': 1.0,
    'lambda_l2': 1.0,
    'verbose': -1,
    'num_threads': 4,
}
xgb_params = {
    'max_depth': 8, 
    'eta': 0.05, 
    'objective': 'binary:logistic',
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'alpha': 1,
    'lambda': 1,
    'eval_metric': 'auc',
    'nthread': 4,
}

cols_to_drop = [
    id_col,
    target_col,
    # 'patient_id',
]

# cols_to_use = features
X = train.drop(cols_to_drop, axis=1, errors='ignore')
y = train[target_col].values

X_test = test.drop(cols_to_drop, axis=1, errors='ignore')
id_test = test[id_col].values

# # Feature selection
# cols_to_drop = []
# for c in X.columns:
#     # t = ttest_ind(
#     #     X[c].fillna(X[c].mean()), 
#     #     X_test[c].fillna(X_test[c].mean()))
#     t = ks_2samp(
#         X[c].dropna(), 
#         X_test[c].dropna())
#     # print(c, t)
#     if t[1] < 0.001:
#         print(c, t)
#         cols_to_drop.append(c)
# print(f'Dropping after statistical tests: {cols_to_drop}')
# X = X.drop(cols_to_drop, axis=1, errors='ignore')
# X_test = X_test.drop(cols_to_drop, axis=1, errors='ignore')

p_test = []
for fold_i, (train_index, valid_index) in enumerate(kf.split(X, y)):
    x_train = X.iloc[train_index].copy()
    x_valid = X.iloc[valid_index].copy()

    y_train = y[train_index]
    y_valid = y[valid_index]

    x_test = X_test.copy()

    # # Frequency encoding
    # encoding = x_train.groupby('height').size()
    # encoding = encoding/len(x_train)
    # x_train['height_fenc'] = x_train['height'].map(encoding)
    # x_valid['height_fenc'] = x_valid['height'].map(encoding)
    # x_test['height_fenc'] = x_test['height'].map(encoding)

    # # Target encoding
    # for c in ['ethnicity', 'gender', 'hospital_admit_source', 'icu_admit_source', 'icu_stay_type', 'icu_type', 'apache_3j_bodysystem', 'apache_2_bodysystem', \
    #     'hospital_id', 'icu_id', 'age_group', 'apache_3j_diagnosis']:
    #     if c in x_train.columns:
    #         trn, sub = target_encode(x_train[c], 
    #             x_valid[c], 
    #             target=train.iloc[train_index][target_col], 
    #             min_samples_leaf=100,
    #             smoothing=10,
    #             noise_level=0.001)
    #         # x_train[c + '_te'] = trn
    #         # x_valid[c + '_te'] = sub
    #         x_valid[c] = sub

    #         trn, sub = target_encode(x_train[c], 
    #             x_test[c], 
    #             target=train.iloc[train_index][target_col], 
    #             min_samples_leaf=100,
    #             smoothing=10,
    #             noise_level=0.001)
    #         # x_test[c + '_te'] = sub
    #         x_train[c] = trn
    #         x_test[c] = sub

    feature_names = list(x_train.columns)  
    n_features = x_train.shape[1]
    dprint(f'n_features: {n_features}')
    p_valid = []

    # LGBM
    if train_lgbm:
        params = lgb_params.copy() 

        # pca = PCA(n_components=144)

        # x_train = pca.fit_transform(x_train)
        # x_valid = pca.transform(x_valid)
        # x_test_pca = pca.transform(x_test)
        # feature_names = ['pca_{}'.format(i) for i in range(x_train.shape[1])]

        lgb_train = lgb.Dataset(
            x_train, 
            y_train, 
            feature_name=feature_names,
            )
        lgb_train.raw_data = None

        lgb_valid = lgb.Dataset(
            x_valid, 
            y_valid,
            )
        lgb_valid.raw_data = None

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=5000,
            valid_sets=[lgb_valid],
            early_stopping_rounds=100,
            verbose_eval=100,
        )

        if fold_i == 0:
            importance = model.feature_importance()
            model_fnames = model.feature_name()
            tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
            tuples = [x for x in tuples if x[1] > 0]
            print('Important features:')
            for i in range(20):
                if i < len(tuples):
                    print(tuples[i])
                else:
                    break

            del importance, model_fnames, tuples

        p_lgbm = model.predict(x_valid, num_iteration=model.best_iteration)
        p_valid.append(p_lgbm)

        err = roc_auc_score(y_valid, p_lgbm)
        # err_buf.append(err)
        dprint('{} LGBM AUC: {:.4f}'.format(fold_i, err))

        p_lgbm_test = model.predict(x_test[feature_names], num_iteration=model.best_iteration)
        p_test.append(p_lgbm_test)

    # XGB
    if train_xgb:
        params = xgb_params.copy() 

        dtrain = xgb.DMatrix(x_train, label=y_train)
        dvalid = xgb.DMatrix(x_valid, label=y_valid)
        dtest = xgb.DMatrix(x_test[feature_names])
        evallist = [(dvalid, 'eval')]

        bst = xgb.train(
            params, 
            dtrain, 
            5000, 
            evallist,
            early_stopping_rounds=100,
            verbose_eval=100
            )

        p_xgb = bst.predict(dvalid, ntree_limit=bst.best_iteration)
        p_valid.append(p_xgb)

        err = roc_auc_score(y_valid, p_xgb)
        # err_buf.append(err)
        dprint('{} XGB AUC: {:.4f}'.format(fold_i, err))

        p_xgb_test = bst.predict(dtest, ntree_limit=bst.best_iteration)
        p_test.append(p_xgb_test)

    # Ensemble evaluation
    if len(p_valid) > 1:
        p_ens = np.mean(p_valid, axis=0)
        err = roc_auc_score(y[valid_index], p_ens)
        dprint('{} ENS AUC: {:.4f}'.format(fold_i, err))
    err_buf.append(err)

    # x_train = X.iloc[train_index]
    # x_valid = X.iloc[valid_index]
    
    # model = NuSVC(
    #     probability=True, 
    #     kernel='poly', 
    #     degree=4, 
    #     gamma='auto', 
    #     random_state=0, 
    #     nu=0.6, 
    #     coef0=0.05)
    # model.fit(x_train, y[train_index])

    # p_nusvc = model.predict_proba(x_valid)[:, 1]
    # err = roc_auc_score(y[valid_index], p_nusvc)
    # print('{} {} NuSVC AUC: {}'.format(v, cnt + 1, err))
    
    # p_nusvc_test = model.predict_proba(x_test)[:, 1]
    
    # p_mean = 0.1*p_lgbm + 0.9*p_nusvc
    # err = roc_auc_score(y[valid_index], p_mean)
    # print('{} {} ENS AUC: {}'.format(v, cnt + 1, err))
    
    # p = 0.1*p_lgbm_test + 0.9*p_nusvc_test

    del model, lgb_train, lgb_valid
    gc.collect

    # break


err_mean = np.mean(err_buf)
err_std = np.std(err_buf)
dprint('AUC: {:.4f} +/- {:.4f}'.format(err_mean, err_std))

test_preds = np.mean(p_test, axis=0)
    
submission = pd.DataFrame()
submission[id_col] = id_test
submission[target_col] = test_preds
submission.to_csv('submission{}.csv'.format(script_id), index=False)

# Save backup
files = [
    'model{}.py'.format(script_id), 
    'model{}.log'.format(script_id), 
    'submission{}.csv'.format(script_id),
    # 'feature_importance{}.txt'.format(script_id),
    # 'train_weights{}.csv'.format(script_id),
]

experiment_name = 'Exp{}'.format(script_id)
params = {}
params['n_models'] = cnt
scores = {}
scores['auc_mean'] = err_mean
scores['auc_std'] = err_std
scores['kaggle'] = np.nan
other = {}
other['n_features'] = n_features
other['n_splits'] = n_splits
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
dprint('Done!')
