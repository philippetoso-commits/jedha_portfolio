"""
Wrapper Ultimate Mix pour intégration dans VotingClassifier
Compatible sklearn avec tous les tags requis
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


class UltimateMixClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper pour Ultimate Mix (Poisson + LogLoss) compatible avec sklearn"""
    
    # CRITIQUE : sklearn vérifie cet attribut pour is_classifier()
    _estimator_type = "classifier"
    
    def __init__(self, learning_rate=0.05, max_iter=500, max_depth=8, 
                 l2_regularization=0.1, random_state=42):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        # Forcer l'attribut d'instance aussi (double sécurité)
        self._estimator_type = "classifier"
    
    def __sklearn_tags__(self):
        """Tags sklearn - CRITIQUE pour is_classifier()"""
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags
        
    def fit(self, X, y):
        """Entraîne les deux modèles (Poisson + LogLoss)"""
        # Définir les classes (requis pour sklearn)
        self.classes_ = np.unique(y)
        
        # Identifier les colonnes catégorielles
        cat_cols = ['country', 'source']
        if hasattr(X, 'columns'):
            cat_indices = [list(X.columns).index(col) for col in cat_cols if col in X.columns]
        else:
            cat_indices = []
        
        # Modèle Poisson (régression)
        self.model_poi_ = HistGradientBoostingRegressor(
            loss='poisson',
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            l2_regularization=self.l2_regularization,
            categorical_features=cat_indices if cat_indices else 'from_dtype',
            random_state=self.random_state
        )
        self.model_poi_.fit(X, y)
        
        # Modèle LogLoss (classification)
        self.model_clf_ = HistGradientBoostingClassifier(
            loss='log_loss',
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            l2_regularization=self.l2_regularization,
            categorical_features=cat_indices if cat_indices else 'from_dtype',
            random_state=self.random_state
        )
        self.model_clf_.fit(X, y)
        
        return self
    
    def predict_proba(self, X):
        """Retourne les probabilités moyennées des deux modèles"""
        # Prédictions Poisson (régression)
        pred_poi = self.model_poi_.predict(X)
        
        # Prédictions LogLoss (classification)
        pred_clf = self.model_clf_.predict_proba(X)[:, 1]
        
        # Moyenne des deux approches
        avg_pred = (pred_poi + pred_clf) / 2
        
        # Retourner les probabilités pour les deux classes
        proba_class_1 = np.clip(avg_pred, 0, 1)
        proba_class_0 = 1 - proba_class_1
        
        return np.column_stack([proba_class_0, proba_class_1])
    
    def predict(self, X):
        """Retourne les prédictions binaires"""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
