from config import *
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
import shap
from lightgbm import LGBMClassifier
from typing import Optional, List, Any

def convert_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    '''Приводит типы данных к меньшей размерности'''
    types_map = {"int": np.int32, "float": np.float32, "object": "category"}
    data = data.copy(deep=True)
    for dtype in ["int", "float", "object"]:
        features = data.select_dtypes(include=dtype).columns
        data[features] = data[features].astype(types_map.get(dtype))
    return data

class ModelTrainer:
    '''Интерфейс для обучения модели и анализа результатов процесса обучения'''
    
    def __init__(self, drop_features: Optional[List[str]] = None) -> None:
        self.drop_features = drop_features or []
        self.results = {}
        
    def train(self, model: Any, 
              X_train: pd.DataFrame, y_train: pd.Series, 
              X_oos: pd.DataFrame, y_oos: pd.Series, 
              X_test: pd.DataFrame, 
              cv: Any):
    
        self.feature_names = X_train.columns.drop(['SK_ID_CURR'] + self.drop_features).to_list()
        self.n_folds = cv.n_splits
        self._initialize_arrays(X_train, X_oos, X_test)

        self.glob_model = clone(model)
        self.glob_model.fit(X_train, y_train)
        
        for train_idx, val_idx in cv.split(X_train, y_train):
            fold_results = self._train_fold(
                clone(model), 
                X_train, y_train, 
                X_oos, y_oos, 
                X_test,
                train_idx, val_idx
            )
            
            self._update_arrays(fold_results, val_idx)
        
        feature_importance_df = self._create_feature_importance_df()
        metrics_df = self._create_metrics_df(y_train, y_oos)
        submission = self._create_submission_df(X_test)
        
        self.results = {
            'train_predictions': np.concatenate(self.train_predictions), 
            'train_labels': np.concatenate(self.train_labels),
            'oof_predictions': self.valid_predictions,          
            'oof_labels': y_train,                     
            'oos_predictions': self.oos_predictions,              
            'oos_labels': y_oos,                       
            'test': self.test_predictions,
            'metrics': metrics_df,
            'feature_importances': feature_importance_df,
            'submission': submission
        }
            
        print("\033[92mTraining has been completed successfully\033[0m")
    
    def _initialize_arrays(self, X_train: pd.DataFrame, X_oos: pd.DataFrame, X_test: pd.DataFrame) -> None:
        self.valid_predictions = np.zeros(X_train.shape[0])
        self.oos_predictions = np.zeros(X_oos.shape[0])
        self.test_predictions = np.zeros(X_test.shape[0])
        
        self.train_predictions = [] 
        self.train_labels = []
        
        self.scores = {'train_roc': [], 'valid_roc': [], 'oos_roc': [],
                      'train_pr': [], 'valid_pr': [], 'oos_pr': []}
        
        self.feature_importance_values = np.zeros(len(self.feature_names))
    
    def _train_fold(self, model: Any, 
                    X_train: pd.DataFrame, y_train: pd.Series,
                    X_oos: pd.DataFrame, y_oos: pd.Series, 
                    X_test: pd.DataFrame,
                    train_idx: np.ndarray, 
                    val_idx: np.ndarray) -> dict:
        X_train_fold = X_train.iloc[train_idx][self.feature_names]
        X_val_fold = X_train.iloc[val_idx][self.feature_names]
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]

        model.fit(X_train_fold, y_train_fold, 
                  eval_metric='auc',
                  eval_set=[(X_val_fold, y_val_fold), (X_train_fold, y_train_fold)], 
                  eval_names=['valid', 'train'])

        train_pred = model.predict_proba(X_train_fold)[:, 1]
        valid_pred = model.predict_proba(X_val_fold)[:, 1]
        oos_pred = model.predict_proba(X_oos[self.feature_names])[:, 1]
        test_pred = model.predict_proba(X_test[self.feature_names])[:, 1]
        
        train_roc_auc = roc_auc_score(y_train_fold, train_pred)
        valid_roc_auc = roc_auc_score(y_val_fold, valid_pred)
        oos_roc_auc = roc_auc_score(y_oos, oos_pred)
        
        train_pr_auc = average_precision_score(y_train_fold, train_pred)
        valid_pr_auc = average_precision_score(y_val_fold, valid_pred)
        oos_pr_auc = average_precision_score(y_oos, oos_pred)
        
        fold_results = {
            'train_pred': train_pred,
            'train_labels': y_train_fold,
            'valid_pred': valid_pred,
            'oos_pred': oos_pred,
            'test_pred': test_pred,
            'train_roc_auc': train_roc_auc,
            'valid_roc_auc': valid_roc_auc,
            'oos_roc_auc': oos_roc_auc,
            'train_pr_auc': train_pr_auc,
            'valid_pr_auc': valid_pr_auc,
            'oos_pr_auc': oos_pr_auc,
            'feature_importance': model.feature_importances_,
        }
            
        return fold_results
    
    def _update_arrays(self, fold_results: dict, val_idx: np.ndarray) -> None:
        self.train_predictions.append(fold_results['train_pred'])
        self.train_labels.append(fold_results['train_labels'])
        
        self.valid_predictions[val_idx] = fold_results['valid_pred']
        
        self.oos_predictions += fold_results['oos_pred'] / self.n_folds
        self.test_predictions += fold_results['test_pred'] / self.n_folds
        
        self.scores['train_roc'].append(fold_results['train_roc_auc'])
        self.scores['valid_roc'].append(fold_results['valid_roc_auc'])
        self.scores['oos_roc'].append(fold_results['oos_roc_auc'])
        self.scores['train_pr'].append(fold_results['train_pr_auc'])
        self.scores['valid_pr'].append(fold_results['valid_pr_auc'])
        self.scores['oos_pr'].append(fold_results['oos_pr_auc'])
        
        self.feature_importance_values += fold_results['feature_importance'] / self.n_folds
    
    def _create_feature_importance_df(self) -> pd.DataFrame:
        feature_importances = pd.DataFrame({
            'feature': self.feature_names, 
            'importance': self.feature_importance_values
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        feature_importances['importance_ratio'] = feature_importances['importance'] / feature_importances['importance'].sum()
        feature_importances['cumulative_importance'] = feature_importances['importance_ratio'].cumsum()
        return feature_importances
    
    def _create_metrics_df(self, y_train: pd.Series, y_oos: pd.Series) -> pd.DataFrame:
        fold_names = list(range(self.n_folds)) + ['MEAN', 'STD', 'OVERALL']
        
        metrics_data = {
            'fold': fold_names,
            'train_roc_auc': self.scores['train_roc'] + [
                np.mean(self.scores['train_roc']), 
                np.std(self.scores['train_roc']),
                roc_auc_score(np.concatenate(self.train_labels), np.concatenate(self.train_predictions)) 
            ],
            'valid_roc_auc': self.scores['valid_roc'] + [
                np.mean(self.scores['valid_roc']), 
                np.std(self.scores['valid_roc']),
                roc_auc_score(y_train, self.valid_predictions)
            ],
            'oos_roc_auc': self.scores['oos_roc'] + [
                np.mean(self.scores['oos_roc']), 
                np.std(self.scores['oos_roc']),
                roc_auc_score(y_oos, self.oos_predictions)
            ],
            'train_pr_auc': self.scores['train_pr'] + [
                np.mean(self.scores['train_pr']), 
                np.std(self.scores['train_pr']),
                average_precision_score(np.concatenate(self.train_labels), np.concatenate(self.train_predictions)) 
            ],
            'valid_pr_auc': self.scores['valid_pr'] + [
                np.mean(self.scores['valid_pr']), 
                np.std(self.scores['valid_pr']),
                average_precision_score(y_train, self.valid_predictions)
            ],
            'oos_pr_auc': self.scores['oos_pr'] + [
                np.mean(self.scores['oos_pr']), 
                np.std(self.scores['oos_pr']),
                average_precision_score(y_oos, self.oos_predictions)
            ]
        }
        
        return pd.DataFrame(metrics_data)
    
    def _create_submission_df(self, X_test: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            'SK_ID_CURR': X_test['SK_ID_CURR'].values, 
            'TARGET': self.test_predictions
        })

    def plot_eval_metrics(self) -> None:
        if not self.results:
            raise ValueError("Model must be trained first")
        
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        data = [
            ('Train', self.results['train_predictions'], self.results['train_labels'], 'blue'),
            ('Validation', self.results['oof_predictions'], self.results['oof_labels'], 'green'),
            ('Out-of-Sample', self.results['oos_predictions'], self.results['oos_labels'], 'red')
        ]

        for name, probs, labels, color in data:
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc = auc(fpr, tpr)
            ax1.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
        ax1.plot([0, 1], [0, 1], 'k--', lw=2)
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend(loc="lower right")

        for name, probs, labels, color in data:
            precision, recall, _ = precision_recall_curve(labels, probs)
            pr_auc = average_precision_score(labels, probs)
            ax2.plot(recall, precision, color=color, lw=2, label=f'{name} (AP = {pr_auc:.3f})')
        
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend(loc="upper right")

        plt.tight_layout()
        plt.show()

    def plot_feature_importances(self, threshold: float = 0.9) -> None:
        df = self.results['feature_importances']
        
        importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
        threshold_info = f'{importance_index + 1} features required for {threshold:.0%} cumulative importance'
        
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        top_n = min(15, len(df))
        y_pos = np.arange(top_n)
        
        ax1.barh(y_pos, df['importance_ratio'].head(top_n)[::-1], align='center')
        ax1.set_yticks(y_pos)
        labels = [label if len(label) <= 20 else label[:17] + '...'  for label in df['feature'].head(top_n)[::-1]]
        ax1.set_yticklabels(labels)
        ax1.set_xlabel('Importance Ratio')
        ax1.set_title(f'Top {top_n} Features')
        
        ax2.plot(range(len(df)), df['cumulative_importance'], linewidth=2)
        ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'{threshold:.0%} threshold')
        ax2.axvline(x=importance_index, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('Cumulative Importance')
        ax2.set_title('Cumulative Feature Importance')
        ax2.legend()
        
        plt.figtext(0.8, 0.15, threshold_info, ha='center', fontsize=12, 
                   bbox=dict(boxstyle="round, pad=0.5", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.show()

    def plot_shap_values(self, data: pd.DataFrame) -> None:
        data = data.copy()
        data.columns =  [feature if len(feature) <= 20 else feature[:17] + '...'  for feature in data.columns]

        explainer = shap.TreeExplainer(model=self.glob_model, feature_perturbation='tree_path_dependent', model_output='raw')
        shap_values = explainer.shap_values(data)

        shap.summary_plot(shap_values, data, max_display=15, plot_size=(12, 8), show=False)

        plt.title('Top 15 Features')
        plt.tight_layout()
        plt.show()


    def save_predictions(self, filepath: str, dataset: str = 'test') -> None:
        if dataset not in ['oof', 'oos', 'test']:
            raise ValueError('Dataset should be one of: oos, oof, test')
        if dataset == 'test':
            self.results['submission'].to_csv(filepath, index=False)
        elif dataset == 'oof':
            pd.DataFrame({'predictions': self.results['oof_predictions']}).to_csv(filepath, index=False)
        elif dataset == 'oos':
            pd.DataFrame({'predictions': self.results['oos_predictions']}).to_csv(filepath, index=False)

def save_model(model: Any, filename: str, **metadata) -> None:
    model_data = {
        'model': model,
        'metadata': {
            **metadata
        }
    }
    joblib.dump(model_data, filename)

def load_model(filename: str) -> None:
    data = joblib.load(filename)
    return data['model'], data['metadata']

def run_recursive_feature_selection(model: Any, 
                                    X_train: pd.DataFrame, y_train: pd.DataFrame, 
                                    X_oos: pd.DataFrame, y_oos: pd.DataFrame, 
                                    cv: Any, 
                                    min_features: int, 
                                    num_features: pd.Index, cat_features: pd.Index) -> dict:
    '''Не жадная версия backward selection по feature importance скору'''
    result = {}
    num_features = num_features.copy()
    cat_features = cat_features.copy()
    remaining_features = num_features.union(cat_features).to_list()

    while len(remaining_features) >= min_features:
        print('=' * 25)
        print(f'Features remaining: {len(remaining_features)}')
        cat_features = cat_features[cat_features.isin(remaining_features)]

        model_: LGBMClassifier = clone(model)
        model_.fit(X_train[remaining_features], y_train, categorical_feature=cat_features.to_list())
        
        cv_scores = cross_val_score(model, X_train[remaining_features], y_train, cv=cv, scoring='roc_auc', fit_params={'categorical_feature': cat_features.to_list()}, n_jobs=5)
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        oos_score = roc_auc_score(y_oos, model_.predict_proba(X_oos[remaining_features])[:, 1])

        print(f'CV score: {mean_cv_score:.4f}')
        print(f'OOS score: {oos_score:.4f}')

        feature_importance_df = pd.DataFrame()
        feature_importance_df['feature'] = model_.booster_.feature_name()
        feature_importance_df['importance'] = model_.booster_.feature_importance(importance_type='gain')
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

        result[len(remaining_features)] = {'features': list(remaining_features), 
                                           'n_features': len(remaining_features),
                                           'cv_score': mean_cv_score,
                                           'cv_std': std_cv_score,
                                           'oos_score': oos_score}
        
        k = feature_importance_df[feature_importance_df['importance'] > 0].shape[0]
        if k >= 300:
            k = int(np.floor(0.85 * k))
        if k >= 100:
            k = int(np.floor(0.90 * k))
        else:
            k = int(np.floor(0.95 * k))

        remaining_features = feature_importance_df['feature'].to_list()[:k]
        
    return result

def run_permutation_feature_selection(model: Any, 
                                      X_train: pd.DataFrame, y_train: pd.DataFrame, 
                                      X_oos: pd.DataFrame, y_oos: pd.DataFrame, 
                                      cv: Any, 
                                      min_features: int, 
                                      num_features: pd.Index, cat_features: pd.Index) -> dict:
    '''Не жадная версия backward selection по permutation скору'''
    result = {}
    num_features = num_features.copy()
    cat_features = cat_features.copy()
    remaining_features = num_features.union(cat_features).to_list()

    while len(remaining_features) >= min_features:
        print('=' * 25)
        print(f'Features remaining: {len(remaining_features)}')
        cat_features = cat_features[cat_features.isin(remaining_features)]

        model_: LGBMClassifier = clone(model)
        model_.fit(X_train[remaining_features], y_train, categorical_feature=cat_features.to_list())
        
        cv_scores = cross_val_score(model, X_train[remaining_features], y_train, cv=cv, scoring='roc_auc', fit_params={'categorical_feature': cat_features.to_list()}, n_jobs=5)
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        oos_score = roc_auc_score(y_oos, model_.predict_proba(X_oos[remaining_features])[:, 1])

        print(f'CV score: {mean_cv_score:.4f}')
        print(f'OOS score: {oos_score:.4f}')

        permutation_data = []
        cv_ = deepcopy(cv)
        cv_.n_splits = 2
        for train_idx, val_idx in cv_.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train.iloc[train_idx][remaining_features], X_train.iloc[val_idx][remaining_features]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model_: LGBMClassifier = clone(model)
            model_.fit(X_train_fold, y_train_fold, categorical_feature=cat_features.to_list())

            permutation_data_fold = permutation_importance(model_, X_val_fold, y_val_fold, scoring='roc_auc', n_repeats=3, n_jobs=-1, random_state=cfg.seed)['importances_mean']
            permutation_data.append(permutation_data_fold)
        
        permutation_importance_df = pd.DataFrame()
        permutation_importance_df['features'] = model_.booster_.feature_name()
        permutation_importance_df['importance'] = np.mean(permutation_data, axis=0)
        permutation_importance_df = permutation_importance_df.sort_values('importance', ascending=False)

        result[len(remaining_features)] = {'features': list(remaining_features), 
                                           'n_features': len(remaining_features),
                                           'cv_score': mean_cv_score,
                                           'cv_std': std_cv_score,
                                           'oos_score': oos_score}
        
        k = permutation_importance_df.shape[0] - 1
        remaining_features = permutation_importance_df['features'].to_list()[:k]
        
    return result

def bootstrap_mean_difference(data: pd.DataFrame, n_bootstraps: int = 10_000, ci: int = 95) -> dict:
    '''Бутстрап доверительный интервал разности средних'''
    group0 = data['TARGET'].values
    group1 = data['PD'].values
    
    observed_diff = np.mean(group1) - np.mean(group0)
    
    boot_diffs = []
    for _ in range(n_bootstraps):
        boot_group0 = np.random.choice(group0, size=len(group0), replace=True)
        boot_group1 = np.random.choice(group1, size=len(group1), replace=True)

        boot_diff = np.mean(boot_group1) - np.mean(boot_group0)
        boot_diffs.append(boot_diff)
    
    alpha = (100 - ci) / 2
    lower_ci = np.percentile(boot_diffs, alpha)
    upper_ci = np.percentile(boot_diffs, 100 - alpha)
    
    significant = not (lower_ci <= 0 <= upper_ci)
    
    return {
        'observed_diff': observed_diff,
        'ci_lower': lower_ci,
        'ci_upper': upper_ci,
        'significant': significant,
        'confidence_level': ci
    }
