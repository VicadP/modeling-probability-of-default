from config import *
import pandas as pd
import numpy as np
from contextlib import contextmanager
import gc
import time
import warnings
from typing import Generator, Union

warnings.filterwarnings('ignore')

@contextmanager
def timer(title: str) -> Generator:
    '''Выводит время работы блока'''
    t0 = time.time()
    yield
    print(f"{title} - done in {time.time() - t0:.0f}s")

def load_data(path: Union[str, Path]) -> pd.DataFrame:
    '''Загружает данные из файла'''
    df = pd.read_csv(path)
    return df

def mark_train(application_train: pd.DataFrame) -> pd.DataFrame:
    '''Размечает исходный набор для обучения на train и oos части'''
    markup = pd.read_csv("./data/train_markup.csv")[['SK_ID_CURR', 'SAMPLE']]
    application_train = application_train.merge(markup, on='SK_ID_CURR', how='left')
    return application_train

# === Data Cleaning ===
def clean_application_data(application_data: pd.DataFrame) -> pd.DataFrame:
    application_data['CODE_GENDER'] = application_data['CODE_GENDER'].replace('XNA', 'M')
    application_data['ORGANIZATION_TYPE'] = application_data['ORGANIZATION_TYPE'].replace('XNA', np.nan)
    application_data['NAME_INCOME_TYPE'] = application_data['NAME_INCOME_TYPE'].replace('Maternity leave', 'Working')
    application_data['NAME_FAMILY_STATUS'] = application_data['NAME_FAMILY_STATUS'].replace('Unknown', 'Married')
    application_data['DAYS_EMPLOYED'] = application_data['DAYS_EMPLOYED'].replace(365243, np.nan)
    application_data['DAYS_LAST_PHONE_CHANGE']= application_data['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan)
    return application_data

def clean_prev_application_data(application_data: pd.DataFrame) -> pd.DataFrame:
    days_features = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']
    application_data[days_features] = application_data[days_features].replace(365243, np.nan)
    return application_data

def clean_bureau_data(bureau_data: pd.DataFrame) -> pd.DataFrame:
    bureau_data['AMT_CREDIT_SUM_DEBT'] = np.clip(bureau_data['AMT_CREDIT_SUM_DEBT'], 0, np.inf)
    return bureau_data

# === Feature Engineering ===
def generate_application_features(application_data: pd.DataFrame) -> pd.DataFrame:
    features = {
        'AMT_ANNUITY_to_AMT_INCOME_TOTAL': application_data['AMT_ANNUITY'] / (application_data['AMT_INCOME_TOTAL'] + 1),
        'AMT_CREDIT_to_AMT_INCOME_TOTAL': application_data['AMT_CREDIT'] / (application_data['AMT_INCOME_TOTAL'] + 1),
        'CRED_TERM': application_data['AMT_CREDIT'] / (application_data['AMT_ANNUITY'] + 1),
        'AMT_CREDIT_diff_AMT_GOODS_PRICE': application_data['AMT_CREDIT'] - application_data['AMT_GOODS_PRICE'],
        'DAYS_EMPLOYED_to_DAYS_BIRTH': application_data['DAYS_EMPLOYED'] / (application_data['DAYS_BIRTH'] + 1),
        'AMT_INCOME_TOTAL_to_CNT_FAM_MEMBERS': application_data['AMT_INCOME_TOTAL'] / application_data['CNT_FAM_MEMBERS']
    }

    active_debt = pd.read_csv(cfg.paths.bureau).query("CREDIT_ACTIVE == 'Active'").groupby('SK_ID_CURR').agg(ACTIVE_DEBT=('AMT_CREDIT_SUM_DEBT', 'sum')).apply(lambda x: np.clip(x, 0, np.inf))
    application_data = application_data.merge(active_debt, on='SK_ID_CURR', how='left').assign(ACTIVE_DEBT=lambda x: x['ACTIVE_DEBT'].fillna(0.0))
    application_data['DTI'] = (application_data['ACTIVE_DEBT'] + application_data['AMT_CREDIT']) / application_data['AMT_INCOME_TOTAL']
    application_data.drop(columns='ACTIVE_DEBT', inplace=True)
    
    return application_data.assign(**features)

def generate_prev_application_features(application_data: pd.DataFrame) -> pd.DataFrame:
    features = {
        'AMT_CREDIT_to_AMT_APPLICATION': application_data['AMT_CREDIT'] / application_data['AMT_APPLICATION'],
        'AMT_CREDIT_diff_AMT_GOODS_PRICE': application_data['AMT_CREDIT'] - application_data['AMT_GOODS_PRICE'],
        'DAYS_FIRST_DUE_diff_DAYS_LAST_DUE_V1': application_data['DAYS_FIRST_DUE'] - application_data['DAYS_LAST_DUE_1ST_VERSION']
    }
    application_data = application_data.assign(**features)

    approved_apps = application_data.query("NAME_CONTRACT_STATUS == 'Approved'")
    refused_apps = application_data.query("NAME_CONTRACT_STATUS == 'Refused'")

    aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean', 'std'],
        'AMT_CREDIT_to_AMT_APPLICATION': ['min', 'mean'],
        'AMT_GOODS_PRICE': ['mean'],
        'AMT_CREDIT_diff_AMT_GOODS_PRICE': ['max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['min', 'max', 'mean'],
    }

    approved_aggregations = {
        'DAYS_FIRST_DUE': ['max', 'mean'],
        'DAYS_FIRST_DUE_diff_DAYS_LAST_DUE_V1': ['max', 'mean'],
        'DAYS_LAST_DUE_1ST_VERSION': ['max', 'mean'],
        'DAYS_TERMINATION': ['max', 'mean']
    }

    approved_apps = approved_apps.groupby('SK_ID_CURR').agg({**aggregations, **approved_aggregations})
    refused_apps = refused_apps.groupby('SK_ID_CURR').agg(aggregations)

    approved_apps.columns = [f'{f[0]}_{f[1].upper()}_APPROVED' for f in approved_apps.columns]
    refused_apps.columns = [f'{f[0]}_{f[1].upper()}_REFUSED' for f in refused_apps.columns]

    return approved_apps.add_prefix("PREV_APP_"), refused_apps.add_prefix("PREV_APP_")

def generate_bureau_features(bureau_data: pd.DataFrame) -> pd.DataFrame:

    def flatten_aggregate(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        '''Переводит 2D индекс в 1D'''
        df_agg = df.agg(aggregates)
        df_agg.columns = [f'{prefix}{f[0]}_{f[1].upper()}' for f in df_agg.columns]
        return df_agg
    
    basis = bureau_data[['SK_ID_CURR']].drop_duplicates()

    bureau_data["ROW_NUM"] = bureau_data.sort_values(["SK_ID_CURR", "DAYS_CREDIT"], ascending=[True, False]).groupby("SK_ID_CURR").cumcount() + 1
    bureau_data['CREDIT_AGE'] = np.abs(bureau_data['DAYS_CREDIT'] // 30 // 12)

    aggregates = {
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'CREDIT_AGE': ['min', 'max', 'mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean', 'std'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'AMT_CREDIT_SUM': ['min', 'max', 'mean', 'std'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'std']
    }

    aggregates_to_merge = [
        (bureau_data, 'GLOB_'),
        (bureau_data.query("CREDIT_ACTIVE == 'Active'"), 'ACTIVE_'),
        (bureau_data.query("ROW_NUM <= 2"), 'LAST2_'),
        (bureau_data.query("ROW_NUM <= 4"), 'LAST4_'),
        (bureau_data.query("ROW_NUM <= 6"), 'LAST6_')
    ]

    credit_statuses = (
        bureau_data
        .pivot_table(
            index='SK_ID_CURR', 
            columns='CREDIT_ACTIVE', 
            aggfunc='size', 
            fill_value=0
        )
        .add_prefix('CRED_')
    )
    for col in credit_statuses.columns:
        credit_statuses[f'{col}_PCT'] = credit_statuses[col] / credit_statuses.sum(axis=1)
        
    repayment = (
        bureau_data.query("CREDIT_ACTIVE == 'Active'")
        .groupby("SK_ID_CURR")[['AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT']]
        .sum()
        .assign(REPAYMENT=lambda x: 1 - x['AMT_CREDIT_SUM_DEBT'] / x['AMT_CREDIT_SUM'])
        [['REPAYMENT']]
    )

    basis = basis.merge(credit_statuses, on='SK_ID_CURR', how='left').merge(repayment, on='SK_ID_CURR', how='left')
    
    for data, prefix in aggregates_to_merge:
        agg_df = flatten_aggregate(data.groupby('SK_ID_CURR'), prefix)
        basis = basis.merge(agg_df, on='SK_ID_CURR', how='left')
        
    return basis.set_index('SK_ID_CURR').add_prefix("BKI_")

def generate_bureau_bal_features(bureau_bal_data: pd.DataFrame, basis: pd.DataFrame) -> pd.DataFrame:

    bureau_bal_data["ROW_NUM"] = bureau_bal_data.sort_values(["SK_ID_BUREAU", "MONTHS_BALANCE"], ascending=[True, False]).groupby("SK_ID_BUREAU").cumcount() + 1
    bureau_bal_data['STATUS_ENC'] = bureau_bal_data['STATUS'].replace({'X': -1, 'C': 0}).astype(np.int32)

    all_stats = pd.get_dummies(bureau_bal_data[['SK_ID_BUREAU', 'STATUS']], columns=['STATUS'], dtype=np.int32)
    all_stats = all_stats.groupby('SK_ID_BUREAU')[['STATUS_0', 'STATUS_1', 'STATUS_3', 'STATUS_4', 'STATUS_5', 'STATUS_C', 'STATUS_X']].agg(['mean', 'sum'])
    all_stats.columns = [f'{f[0]}_{f[1].upper()}' for f in all_stats.columns]
    
    basis = (
        basis
        .merge(
            all_stats,
            on='SK_ID_BUREAU',
            how='left'
        )
        .merge(
            bureau_bal_data[bureau_bal_data['STATUS_ENC'] >= 1].groupby('SK_ID_BUREAU')[['MONTHS_BALANCE']].min().rename(columns={'MONTHS_BALANCE': 'MONTHS_FROM_DELIQUENCY'}),
            on='SK_ID_BUREAU',
            how='left'
        )
        .merge(
            bureau_bal_data[bureau_bal_data['STATUS_ENC'] >= 3].groupby('SK_ID_BUREAU')[['MONTHS_BALANCE']].min().rename(columns={'MONTHS_BALANCE': 'MONTHS_FROM_DPD90+'}),
            on='SK_ID_BUREAU',
            how='left'
        )
        .merge(
            bureau_bal_data.groupby('SK_ID_BUREAU')['STATUS_ENC'].agg(['max', 'mean']).rename(columns={'max': 'WORST_DELIQENCY_STATUS', 'mean': 'AVG_DELIQUENCY_STATUS'}),
            on='SK_ID_BUREAU',
            how='left'
        )
        .merge(
            bureau_bal_data.query("ROW_NUM <= 1").groupby('SK_ID_BUREAU')['STATUS_ENC'].agg(['max', 'mean']).rename(columns={'max': 'WORST_DELIQENCY_STATUS_1m', 'mean': 'AVG_DELIQUENCY_STATUS_1m'}),
            on='SK_ID_BUREAU',
            how='left'
        )
        .merge(
            bureau_bal_data.query("ROW_NUM <= 3").groupby('SK_ID_BUREAU')['STATUS_ENC'].agg(['max', 'mean']).rename(columns={'max': 'WORST_DELIQENCY_STATUS_3m', 'mean': 'AVG_DELIQUENCY_STATUS_3m'}),
            on='SK_ID_BUREAU',
            how='left'
        )
        .merge(
            bureau_bal_data.query("ROW_NUM <= 6").groupby('SK_ID_BUREAU')['STATUS_ENC'].agg(['max', 'mean']).rename(columns={'max': 'WORST_DELIQENCY_STATUS_6m', 'mean': 'AVG_DELIQUENCY_STATUS_6m'}),
            on='SK_ID_BUREAU',
            how='left'
        )
    )

    basis = basis.drop(columns='SK_ID_BUREAU').groupby('SK_ID_CURR').agg(['min', 'max', 'mean'])
    basis.columns = [f'{f[0]}_{f[1].upper()}' for f in basis.columns]

    return basis.add_prefix("BKI_BB_")

def generate_installments_features(installments_data: pd.DataFrame) -> pd.DataFrame:

    def get_last_n_features(n: int, prefix: str):
        '''Генерирует данные по последним n платежам'''
        last_n = installments_data.query(f"ROW_NUM <= {n}").groupby('SK_ID_CURR').agg(aggregations)
        last_n.columns = [f'{prefix}_{f[0]}_{f[1].upper()}' for f in last_n.columns]
        return last_n
    
    basis = installments_data[['SK_ID_CURR', 'SK_ID_PREV']].drop_duplicates()
    
    payment_aggs = (
        installments_data
        .groupby(['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'])
        .agg(
            CNT_PAYMENTS=('AMT_INSTALMENT', 'count'),
            TOTAL_PAYMENT=('AMT_PAYMENT', 'sum')
        )
    )
    
    installments_data = installments_data.merge(
        payment_aggs, 
        on=['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'], 
        how='left'
    )
    
    installments_data = installments_data.assign(
        PARTIAL_PAYMENT=(installments_data['CNT_PAYMENTS'] > 1).astype(int),
        PAYMENT_DIFF=installments_data['AMT_INSTALMENT'] - installments_data['TOTAL_PAYMENT'],
        PAYMENT_DELTA_DAYS=installments_data['DAYS_INSTALMENT'] - installments_data['DAYS_ENTRY_PAYMENT']
    )
    
    installments_data = installments_data.assign(
        DPD=np.where(installments_data['PAYMENT_DELTA_DAYS'] < 0, -installments_data['PAYMENT_DELTA_DAYS'], 0),
        OVR_PAYMENT=(installments_data['PAYMENT_DELTA_DAYS'] < 0).astype(int),
        ROW_NUM=installments_data.groupby('SK_ID_CURR')['DAYS_INSTALMENT'].rank(method='min', ascending=False)
    )
    
    installments_data = installments_data.drop(columns=['CNT_PAYMENTS', 'TOTAL_PAYMENT'])
    
    aggregations = {
        'NUM_INSTALMENT_NUMBER': ['max', 'mean'],
        'DAYS_INSTALMENT': ['min', 'max', 'mean'],
        'DAYS_ENTRY_PAYMENT': ['min', 'max', 'mean'],
        'AMT_INSTALMENT': ['max', 'mean'],
        'AMT_PAYMENT': ['max', 'mean'],
        'PAYMENT_DIFF': ['max', 'mean', 'std'],
        'PAYMENT_DELTA_DAYS': ['max', 'mean', 'std'],
        'DPD': ['max', 'mean'],
        'OVR_PAYMENT': ['sum', 'mean'],
        'PARTIAL_PAYMENT': ['sum', 'mean']
    }
    
    prev_ids = installments_data.groupby('SK_ID_PREV').agg(aggregations)
    prev_ids.columns = [f'{f[0]}_{f[1].upper()}' for f in prev_ids.columns]
    
    basis = (
        basis
        .merge(prev_ids, on='SK_ID_PREV', how='left')
        .drop(columns='SK_ID_PREV')
        .groupby('SK_ID_CURR')
        .agg(['min', 'max', 'mean'])
    )
    basis.columns = [f'GLOB_{f[0]}_{f[1].upper()}' for f in basis.columns]
    
    for n, prefix in [(2, 'LAST2'), (4, 'LAST4'), (6, 'LAST6')]:
        agg_df = get_last_n_features(n, prefix)
        basis = basis.merge(agg_df, on='SK_ID_CURR', how='left')
    
    return basis.add_prefix("INST_")

def generate_within_client_features(application_data: pd.DataFrame) -> pd.DataFrame:
    application_data['AMT_ANNUITY_to_PREV_APP_AMT_ANNUITY_MAX_APPROVED'] = application_data['AMT_ANNUITY'] / application_data['PREV_APP_AMT_ANNUITY_MAX_APPROVED']
    application_data['AMT_ANNUITY_to_PREV_APP_AMT_ANNUITY_MEAN_APPROVED'] = application_data['AMT_ANNUITY'] / application_data['PREV_APP_AMT_ANNUITY_MEAN_APPROVED']
    application_data['AMT_CREDIT_to_PREV_APP_AMT_CREDIT_MAX_APPROVED'] = application_data['AMT_CREDIT'] / application_data['PREV_APP_AMT_CREDIT_MAX_APPROVED']
    application_data['AMT_CREDIT_to_PREV_APP_AMT_CREDIT_MEAN_APPROVED'] = application_data['AMT_CREDIT'] / application_data['PREV_APP_AMT_CREDIT_MEAN_APPROVED']
    return application_data

def main():
    with timer("Process application data"):
        application_train = load_data(cfg.paths.applic_train).pipe(clean_application_data).pipe(mark_train).pipe(generate_application_features)
        application_test = load_data(cfg.paths.applic_test).pipe(clean_application_data).pipe(generate_application_features)

    with timer("Process previous application data"):
        approved, refused = load_data(cfg.paths.applic_prev).pipe(clean_prev_application_data).pipe(generate_prev_application_features)

        application_train = application_train.merge(approved, on='SK_ID_CURR', how='left').merge(refused, on='SK_ID_CURR', how='left')
        application_test = application_test.merge(approved, on='SK_ID_CURR', how='left').merge(refused, on='SK_ID_CURR', how='left')
    
    del approved, refused
    gc.collect()

    with timer("Process bureau data"):
        basis = load_data(cfg.paths.bureau)[['SK_ID_CURR', 'SK_ID_BUREAU']]
        bureau = load_data(cfg.paths.bureau).pipe(clean_bureau_data).pipe(generate_bureau_features)
        bureau_bal = load_data(cfg.paths.bureau_bal).pipe(generate_bureau_bal_features, basis)

        application_train = application_train.merge(bureau, on='SK_ID_CURR', how='left').merge(bureau_bal, on='SK_ID_CURR', how='left')
        application_test = application_test.merge(bureau, on='SK_ID_CURR', how='left').merge(bureau_bal, on='SK_ID_CURR', how='left')

        del basis, bureau, bureau_bal
        gc.collect()

    with timer("Process installments data"):
        installments = load_data(cfg.paths.installments).pipe(generate_installments_features)

        application_train = application_train.merge(installments, on='SK_ID_CURR', how='left')
        application_test = application_test.merge(installments, on='SK_ID_CURR', how='left')

        del installments
        gc.collect()

    with timer("Process within client data"):
        application_train = generate_within_client_features(application_train)
        application_test = generate_within_client_features(application_test)

    with timer("Save processed data"):
        application_train.to_csv('./data/cleaned/application_train.csv', index=False)
        application_test.to_csv('./data/cleaned/application_test.csv', index=False)

if __name__ == "__main__":
    with timer("Full run"):
        main()