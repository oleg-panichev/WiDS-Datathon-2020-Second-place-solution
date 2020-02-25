import datetime
import gc
import glob
import numpy as np
import os
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK']='True' # MacOS fix for libomp issues (https://github.com/dmlc/xgboost/issues/1715)

import lightgbm as lgb

from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error
from sklearn.model_selection import KFold, RepeatedKFold, GroupKFold, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import NuSVC
from tqdm import tqdm as tqdm

from kinoa import kinoa
from scipy.stats import ttest_ind, ks_2samp


def dprint(*args, **kwargs):
    print("[{}] ".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) + \
        " ".join(map(str,args)), **kwargs)


dprint('PID: {}'.format(os.getpid()))
script_id = 0

data_path = '../input/'

train = pd.read_csv(os.path.join(data_path, 'training_v2.csv'))

id_col = 'encounter_id'

fd = pd.read_csv(os.path.join(data_path, 'WiDS Datathon 2020 Dictionary.csv'))
fd = fd[(fd['Data Type'] == 'string') | (fd['Data Type'] == 'binary')]
cat_features = list(fd['Variable Name'].values)
for c in cat_features:
    if c not in train.columns or c == 'hospital_death':
        cat_features.remove(c)
print(f'cat_features: {cat_features} ({len(cat_features)})')


extracted_files = glob.glob('./*.csv')
extracted_files = [f[2:-8] for f in extracted_files]
print(extracted_files)
# error

target_cols = []
for c in train.columns:
    if c != id_col and c != 'hospital_death' and train[c].isnull().mean() > 0 and c not in extracted_files and c not in cat_features:
        target_cols.append({'fname': c, 'type': 'regression'})

print(target_cols)

def preprocess(df, min_max_cols):
    for c in min_max_cols:
        vals = df[[c, c.replace('_min', '_max')]].values.copy()
        df[c] = np.nanmin(vals, axis=1)
        df[c.replace('_min', '_max')] = np.nanmax(vals, axis=1)

for t_i, target_data in enumerate(target_cols):
    target_col = target_data['fname']
    dprint(f'********************************* {target_col} ({t_i+1}/{len(target_cols)}) *********************************')

    train = pd.read_csv(os.path.join(data_path, 'training_v2.csv'))
    test = pd.read_csv(os.path.join(data_path, 'unlabeled.csv'))

    min_max_cols = []
    for c in train.columns:
        if '_min' in c and c.replace('min', 'max') in train.columns:
            min_max_cols.append(c)
    print(f'min_max_cols: {min_max_cols} ({len(min_max_cols)})')

    preprocess(train, min_max_cols)
    preprocess(test, min_max_cols)

    print(f'Number of missing values in train: {train[target_col].isnull().mean()}')
    print(f'Number of missing values in test: {test[target_col].isnull().mean()}')

    train['is_test'] = 0
    test['is_test'] = 1
    df_all = pd.concat([train, test], axis=0)

    dprint('Label Encoder...')
    cols = [f_ for f_ in df_all.columns if df_all[f_].dtype == 'object']
    print(cols)
    cnt = 0
    for c in tqdm(cols):
        if c != id_col and c != target_col:
            # print(c)
            le = LabelEncoder()
            df_all[c] = le.fit_transform(df_all[c].astype(str))
            cnt += 1

            del le
    dprint('len(cols) = {}'.format(cnt))

    train = df_all.loc[df_all['is_test'] == 0].drop(['is_test'], axis=1)
    test = df_all.loc[df_all['is_test'] == 1].drop(['is_test'], axis=1)

    # del df_all
    # gc.collect()

    # Rearrange train and test
    train = df_all[np.logical_not(df_all[target_col].isnull())].drop(['is_test'], axis=1)
    test = df_all[df_all[target_col].isnull()].drop(['is_test'], axis=1)
    dprint(train.shape, test.shape) 

    if target_data['type'] == 'classification':
        tle = LabelEncoder()
        train[target_col] = tle.fit_transform(train[target_col].astype(str))

    empty_cols = []
    for c in test.columns:
        n = (~test[c].isnull()).sum()
        if n == 0:
            empty_cols.append(c)
    print(f'empty_cols: {empty_cols}')


    # error
    features = list(train.columns.values)
    features.remove(id_col)
    features.remove(target_col)


    # Build the model
    cnt = 0
    p_buf = []
    n_splits = 4
    n_repeats = 1
    kf1 = RepeatedKFold(
        n_splits=n_splits, 
        n_repeats=n_repeats, 
        random_state=0)
    kf2 = RepeatedKFold(
        n_splits=n_splits, 
        n_repeats=n_repeats, 
        random_state=1)
    err_buf = []   
    undersampling = 0

    if target_data['type'] == 'regression':
        lgb_params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'mse',
            'max_depth': 8,
            'learning_rate': 0.05, 
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 5,
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
            'verbose': -1,
            'num_threads': -1,
        }
    elif target_data['type'] == 'classification':
        dprint(f'Num classes: {train[target_col].nunique()} ({train[target_col].unique()})')
        if train[target_col].nunique() == 2:
            lgb_params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'binary_logloss',
                'max_depth': 8,
                'learning_rate': 0.05, 
                'feature_fraction': 0.85,
                'bagging_fraction': 0.85,
                'bagging_freq': 5,
                'lambda_l1': 1.0,
                'lambda_l2': 1.0,
                'verbose': -1,
                'num_threads': -1,
            }
        else:
            lgb_params = {
                'boosting_type': 'gbdt',
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'max_depth': 8,
                'learning_rate': 0.05, 
                'feature_fraction': 0.85,
                'bagging_fraction': 0.85,
                'bagging_freq': 5,
                'lambda_l1': 1.0,
                'lambda_l2': 1.0,
                'verbose': -1,
                'num_threads': -1,
                'num_class': train[target_col].nunique()
            }

    cols_to_drop = [
        id_col,
        target_col,
        'hospital_death',
        # 'bmi',
    ] + empty_cols

    # cols_to_use = features
    X = train.drop(cols_to_drop, axis=1, errors='ignore')
    y = train[target_col].values
    id_train = train[id_col].values

    X_test = test.drop(cols_to_drop, axis=1, errors='ignore')
    id_test = test[id_col].values

    feature_names = list(X.columns)
    n_features = X.shape[1]
    dprint(f'n_features: {n_features}')

    p_test = []
    dfs_train = []
    dfs_test = []

    for fold_i_oof, (train_index_oof, valid_index_oof) in enumerate(kf1.split(X, y)):
        x_train_oof = X.iloc[train_index_oof]
        x_valid_oof = X.iloc[valid_index_oof]

        y_train_oof = y[train_index_oof]
        y_valid_oof = y[valid_index_oof]

        id_train_oof = id_train[valid_index_oof]

        for fold_i, (train_index, valid_index) in enumerate(kf2.split(x_train_oof, y_train_oof)):
            params = lgb_params.copy() 

            x_train = x_train_oof.iloc[train_index]
            x_valid = x_train_oof.iloc[valid_index]

            lgb_train = lgb.Dataset(
                x_train, 
                y_train_oof[train_index], 
                feature_name=feature_names,
                )
            lgb_train.raw_data = None

            lgb_valid = lgb.Dataset(
                x_valid, 
                y_train_oof[valid_index],
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

            if fold_i_oof == 0:
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
            if target_data['type'] == 'regression':
                err = mean_squared_error(y_train_oof[valid_index], p_lgbm)
                err_buf.append(err)
                dprint('{} LGBM MSE: {:.4f}'.format(fold_i, err))
            elif target_data['type'] == 'classification':
                if train[target_col].nunique() == 2:
                    err = roc_auc_score(y_train_oof[valid_index], p_lgbm)
                    dprint('{} LGBM AUC: {:.6f}'.format(fold_i, err))
                err = log_loss(y_train_oof[valid_index], p_lgbm)
                err_buf.append(err)
                dprint('{} LGBM LOSS: {:.4f}'.format(fold_i, err))

            p_lgbm_train = model.predict(x_valid_oof, num_iteration=model.best_iteration)
            p_lgbm_test = model.predict(X_test[feature_names], num_iteration=model.best_iteration)

            df_train = pd.DataFrame()
            df_train[id_col] = id_train_oof
            if target_data['type'] == 'regression':
                df_train[target_col] = p_lgbm_train
            elif target_data['type'] == 'classification':
                if train[target_col].nunique() == 2:
                    df_train[target_col] = p_lgbm_train
                else:
                    for i, t in enumerate(np.sort(train[target_col].unique())):
                        df_train[str(t)] = p_lgbm_train[:, i]

            dfs_train.append(df_train)

            df_test = pd.DataFrame()
            df_test[id_col] = id_test
            if target_data['type'] == 'regression':
                df_test[target_col] = p_lgbm_test
            elif target_data['type'] == 'classification':
                if train[target_col].nunique() == 2:
                    df_test[target_col] = p_lgbm_test
                else:
                    for i, t in enumerate(np.sort(train[target_col].unique())):
                        df_test[str(t)] = p_lgbm_test[:, i]

            dfs_test.append(df_test)
            
            # p_test.append(p_lgbm_test)

            del model, lgb_train, lgb_valid
            gc.collect

        # break

    err_mean = np.mean(err_buf)
    err_std = np.std(err_buf)
    dprint('ERR: {:.4f} +/- {:.4f}'.format(err_mean, err_std))

    dfs_train = pd.concat(dfs_train, axis=0)
    if target_data['type'] == 'regression':
        dfs_train = dfs_train.groupby(id_col)[target_col].mean().reset_index().rename({target_col: target_col + '_est'}, axis=1)
    elif target_data['type'] == 'classification':
        if train[target_col].nunique() == 2:
            dfs_train = dfs_train.groupby(id_col)[target_col].mean().reset_index()
            dfs_train[target_col] = tle.inverse_transform(np.round(dfs_train[target_col].values).astype(int))
            dfs_train.rename({target_col: target_col + '_est'}, inplace=True, axis=1)
        else:
            dfs_train = dfs_train.groupby(id_col).mean().reset_index()
            cols = np.sort(train[target_col].unique()).astype(str)
            dfs_train[target_col + '_est'] = tle.inverse_transform(np.argmax(dfs_train[cols].values, axis=1))
    print(dfs_train.head())

    dfs_test = pd.concat(dfs_test, axis=0)
    if target_data['type'] == 'regression':
        dfs_test = dfs_test.groupby(id_col)[target_col].mean().reset_index().rename({target_col: target_col + '_est'}, axis=1)
    elif target_data['type'] == 'classification':
        if train[target_col].nunique() == 2:
            dfs_test = dfs_test.groupby(id_col)[target_col].mean().reset_index()
            dfs_test[target_col] = tle.inverse_transform(np.round(dfs_test[target_col].values).astype(int))
            dfs_test.rename({target_col: target_col + '_est'}, inplace=True, axis=1)
        else:
            dfs_test = dfs_test.groupby(id_col).mean().reset_index()
            cols = np.sort(train[target_col].unique()).astype(str)
            dfs_test[target_col + '_est'] = tle.inverse_transform(np.argmax(dfs_test[cols].values, axis=1))
    print(dfs_test.head())

    out = pd.concat([dfs_train, dfs_test], axis=0)
    out.to_csv(target_col + '_est.csv', index=False)
