from config import *
from utils import convert_dtypes, save_model
import pandas as pd
from datetime import datetime
from lightgbm import LGBMClassifier

def main(feature_set, cv_score=None, lb_score=None) -> None:
    train_data = pd.read_csv(cfg.paths.train_sample).pipe(convert_dtypes)
    train = train_data.query("SAMPLE == 'TRAIN'")
    X_train = train[feature_set]
    y_train = train['TARGET']

    model = LGBMClassifier(
        **cfg.training.model_params,
        verbosity=-1, 
        importance_type='gain', 
        random_state=cfg.seed, 
    )

    model.fit(X_train, y_train)

    metadata = {
        'creator': 'Ponomarev Viktor',
        'created_at': datetime.today().strftime('%Y-%m-%d'),
        'data_sources': ['applications', 'previous applications', 'bureau', 'installments'],
        'training_samples': X_train.shape[0],
        'n_features': X_train.shape[1],
        'features': feature_set,
        'cv_score': cv_score,
        'lb_score': lb_score
    }

    save_model(model, 'lgbm_shortlist.joblib', **metadata)

    predict = model.predict_proba(train_data[feature_set])[:, 1]
    train_data['PD'] = predict
    train_data.to_csv('./data/scored/application_train.csv', index=False)

if __name__ == '__main__':
    main(feature_set=cfg.training.short_list)