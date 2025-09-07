from pathlib import Path
from easydict import EasyDict

cfg = EasyDict()
cfg.seed = 42

cfg.training = EasyDict()
cfg.training.model_params = ({
 'boosting_type': 'gbdt', 
 'objective': 'binary', 
 'n_estimators': 250,
 'learning_rate': 0.05, 
 'num_leaves': 27,
 'min_child_samples': 1857,
 'colsample_bytree': 0.5002287019819426,
 'reg_alpha': 0.019998333098629646,
 'reg_lambda': 2.6663467952723607e-06
})
cfg.training.short_list = ([
    'EXT_SOURCE_2',
    'EXT_SOURCE_3',
    'EXT_SOURCE_1',
    'AMT_CREDIT_diff_AMT_GOODS_PRICE',
    'CRED_TERM',
    'CODE_GENDER',
    'DAYS_BIRTH',
    'PREV_APP_DAYS_LAST_DUE_1ST_VERSION_MAX_APPROVED',
    'ORGANIZATION_TYPE',
    'DAYS_EMPLOYED',
    'BKI_REPAYMENT',
    'OWN_CAR_AGE',
    'INST_LAST6_DPD_MEAN',
    'AMT_GOODS_PRICE',
    'INST_GLOB_DAYS_INSTALMENT_MIN_MIN',
    'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS',
    'PREV_APP_DAYS_FIRST_DUE_diff_DAYS_LAST_DUE_V1_MEAN_APPROVED',
    'BKI_CRED_Closed_PCT',
    'PREV_APP_AMT_CREDIT_to_AMT_APPLICATION_MIN_APPROVED',
    'DAYS_ID_PUBLISH',
    'INST_GLOB_DPD_MAX_MEAN',
    'AMT_ANNUITY',
    'AMT_ANNUITY_to_PREV_APP_AMT_ANNUITY_MAX_APPROVED',
    'FLAG_DOCUMENT_3',
    'PREV_APP_DAYS_TERMINATION_MAX_APPROVED',
    'BKI_GLOB_DAYS_CREDIT_MAX',
    'INST_GLOB_OVR_PAYMENT_MEAN_MEAN',
    'INST_LAST4_PAYMENT_DELTA_DAYS_MEAN',
    'PREV_APP_AMT_CREDIT_MIN_APPROVED',
    'PREV_APP_DAYS_DECISION_MEAN_REFUSED',
    'BKI_GLOB_AMT_CREDIT_MAX_OVERDUE_MAX',
    'INST_LAST2_AMT_INSTALMENT_MAX',
    'AMT_ANNUITY_to_PREV_APP_AMT_ANNUITY_MEAN_APPROVED',
    'BKI_LAST2_AMT_CREDIT_SUM_DEBT_STD',
    'OCCUPATION_TYPE',
    'INST_LAST6_DAYS_ENTRY_PAYMENT_MIN',
    'INST_LAST2_AMT_PAYMENT_MEAN',
    'BKI_ACTIVE_DAYS_CREDIT_MAX',
    'PREV_APP_DAYS_LAST_DUE_1ST_VERSION_MEAN_APPROVED',
    'REGION_RATING_CLIENT_W_CITY',
    'BKI_CRED_Active',
    'BKI_LAST2_AMT_CREDIT_SUM_DEBT_MEAN',
    'DEF_30_CNT_SOCIAL_CIRCLE',
    'BKI_LAST2_AMT_CREDIT_SUM_MIN',
    'BKI_ACTIVE_AMT_CREDIT_SUM_STD',
    'INST_GLOB_PAYMENT_DELTA_DAYS_STD_MIN',
    'REGION_POPULATION_RELATIVE',
    'INST_GLOB_NUM_INSTALMENT_NUMBER_MEAN_MAX',
    'BKI_ACTIVE_AMT_CREDIT_SUM_MAX',
    'PREV_APP_CNT_PAYMENT_MAX_REFUSED',
    'DTI',
    'INST_LAST4_DAYS_ENTRY_PAYMENT_MIN',
    'PREV_APP_AMT_CREDIT_STD_REFUSED',
    'BKI_GLOB_AMT_CREDIT_SUM_MEAN',
    'INST_LAST6_AMT_PAYMENT_MEAN',
    'INST_GLOB_DAYS_ENTRY_PAYMENT_MIN_MEAN',
    'INST_LAST2_DAYS_ENTRY_PAYMENT_MAX',
    'INST_GLOB_NUM_INSTALMENT_NUMBER_MAX_MAX',
    'PREV_APP_AMT_CREDIT_to_AMT_APPLICATION_MEAN_APPROVED',
    'BKI_LAST2_AMT_CREDIT_MAX_OVERDUE_MEAN',
    'INST_LAST2_PAYMENT_DELTA_DAYS_MEAN',
    'AMT_ANNUITY_to_AMT_INCOME_TOTAL',
    'BKI_LAST6_DAYS_CREDIT_MEAN'
])

cfg.paths = EasyDict()
cfg.paths.root = Path().cwd()
cfg.paths.applic_train = cfg.paths.root / "data/application_train.csv"
cfg.paths.applic_test = cfg.paths.root / "data/application_test.csv"
cfg.paths.applic_prev = cfg.paths.root / "data/previous_application.csv"
cfg.paths.bureau = cfg.paths.root / "data/bureau.csv"
cfg.paths.bureau_bal = cfg.paths.root / "data/bureau_balance.csv"
cfg.paths.installments = cfg.paths.root / "data/installments_payments.csv"
cfg.paths.train_sample = cfg.paths.root / "data/cleaned/application_train.csv"
cfg.paths.test_sample = cfg.paths.root / "data/cleaned/application_test.csv"
cfg.paths.train_sample_scored = cfg.paths.root / "data/scored/application_train.csv"