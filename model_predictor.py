"""
Model Predictor Module for Carrefour Voyages Simulator
Loads the best trained model and provides prediction functions.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

class CarrefourPredictor:
    def __init__(self):
        self.model = None
        self.model_path = None
        
    def load_model(self):
        """Load the best trained model (try ultimate first, fallback to best)"""
        script_dir = Path(__file__).resolve().parent
        
        # Try ultimate model first
        ultimate_path = script_dir.parent.joinpath('03_RESULTATS_ANALYSE', 'models', 'ultimate_model.joblib')
        if ultimate_path.exists():
            self.model_path = ultimate_path
            self.model = joblib.load(self.model_path)
            return self
        
        # Fallback to best model
        best_path = script_dir.parent.joinpath('03_RESULTATS_ANALYSE', 'models', 'best_model.joblib')
        if best_path.exists():
            self.model_path = best_path
            self.model = joblib.load(self.model_path)
            return self
        
        raise FileNotFoundError(f"No model found in models directory")
    
    
    def prepare_features(self, input_dict):
        """
        Prepare features from user inputs for prediction.
        Now supports ULTIMATE MODEL with 28 features.
        """
        # Create DataFrame with expected features
        df = pd.DataFrame([input_dict])
        
        # Add engineered features (original)
        df['CA_Per_Effectif_Ratio_Web'] = df['CA_Par_Effectif'] * df['Ratio_Web']
        df['Effectifs_CDI_Ratio_Manager'] = df['Effectifs_CDI'] * df['Ratio_Manager']
        df['Log_Nombre_Resa'] = np.log1p(df['Nombre_Resa_Total'])
        
        # Add ULTIMATE features
        df['Log_Effectifs'] = np.log1p(df['Effectifs_Totaux'])
        df['Log_CA_Par_Effectif'] = np.log1p(df['CA_Par_Effectif'])
        df['Resa_Per_Effectif'] = df['Nombre_Resa_Total'] / df['Effectifs_Totaux'].replace(0, 1)
        df['Log_Resa_Per_Effectif'] = np.log1p(df['Resa_Per_Effectif'])
        df['Productivite_Web'] = df['CA_Par_Effectif'] * df['Ratio_Web']
        df['Experience_Stabilite'] = df['Anciennete_moyenne_annees'] * df['Ratio_CDI']
        df['Taille_Experience'] = df['Effectifs_Totaux'] * df['Anciennete_moyenne_annees']
        df['Manager_Anciennete'] = df['Ratio_Manager'] * df['Anciennete_moyenne_annees']
        df['CDI_vs_CDD'] = df['Effectifs_CDI'] / df.get('Effectifs_CDD', 1).replace(0, 1)
        df['TempsPlein_vs_Partiel'] = df['Effectif_Temps_Plein'] / df.get('Effectif_Temps_Partiel', 1).replace(0, 1)
        df['Effectifs_Squared'] = df['Effectifs_Totaux'] ** 2
        df['Log_Resa_Squared'] = df['Log_Nombre_Resa'] ** 2
        df['Ratio_Web_Squared'] = df['Ratio_Web'] ** 2
        
        # Select features in order expected by ULTIMATE model (28 features)
        features = [
            'Effectifs_Totaux', 'Anciennete_moyenne_annees',
            'Effectifs_CDI', 'Effectifs_CDD', 'Effectifs_Aternance',
            'Effectif_Responsable_Agence', 'Effectif_Temps_Plein', 'Effectif_Temps_Partiel',
            'Ratio_CDI', 'Ratio_Temps_Plein', 'Ratio_Manager', 'Ratio_Web',
            'Log_Nombre_Resa', 'Log_Effectifs', 'Log_CA_Par_Effectif',
            'CA_Per_Effectif_Ratio_Web', 'Effectifs_CDI_Ratio_Manager',
            'Resa_Per_Effectif', 'Log_Resa_Per_Effectif',
            'Productivite_Web', 'Experience_Stabilite', 'Taille_Experience', 'Manager_Anciennete',
            'CDI_vs_CDD', 'TempsPlein_vs_Partiel',
            'Effectifs_Squared', 'Log_Resa_Squared', 'Ratio_Web_Squared'
        ]
        
        return df[features].fillna(0).replace([np.inf, -np.inf], 0)
    
    def predict_ca(self, input_dict):
        """
        Predict CA_Total from input features.
        Returns predicted CA (original scale).
        """
        if self.model is None:
            self.load_model()
        
        X = self.prepare_features(input_dict)
        # Model was trained on log1p(y), so we need to inverse transform
        y_pred_log = self.model.predict(X)
        y_pred = np.expm1(y_pred_log)
        
        return float(y_pred[0])
    
    def predict_with_scenarios(self, base_input, scenarios):
        """
        Predict CA for multiple scenarios.
        
        scenarios: list of dicts with keys to modify from base_input
        Returns: list of (scenario_name, predicted_ca) tuples
        """
        results = []
        for scenario_name, modifications in scenarios.items():
            scenario_input = base_input.copy()
            scenario_input.update(modifications)
            predicted_ca = self.predict_ca(scenario_input)
            results.append((scenario_name, predicted_ca))
        
        return results


def calculate_cluster_simple(ca_total, nombre_resa_total, effectifs_totaux):
    """
    Simple cluster assignment based on performance quantiles.
    More robust than using the full clustering model.
    
    Returns cluster number (0-4) where higher = better performance
    """
    # Calculate key metrics
    ca_per_effectif = ca_total / max(effectifs_totaux, 1)
    resa_per_effectif = nombre_resa_total / max(effectifs_totaux, 1)
    
    # Performance score (weighted combination)
    performance_score = (
        0.6 * ca_per_effectif / 100000 +  # Normalize CA per effectif
        0.4 * resa_per_effectif / 100      # Normalize resa per effectif
    )
    
    # Assign cluster based on performance quartiles
    if performance_score < 1.0:
        return 0  # Low performance
    elif performance_score < 2.0:
        return 1  # Below average
    elif performance_score < 3.5:
        return 2  # Average
    elif performance_score < 5.0:
        return 3  # Good
    else:
        return 4  # Excellent
