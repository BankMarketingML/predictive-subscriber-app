# model_utils.py

import shap
import matplotlib.pyplot as plt
import os
import streamlit as st
from sklearn.metrics import roc_curve, auc

def plot_roc(model, X, y):
    """
    Génère et sauvegarde la courbe ROC pour un modèle donné

    Args:
        model: Le modèle entraîné (pipeline ou estimator)
        X: Données de test (features)
        y: Données de test (target)
    """
    try:
        st.info("Génération de la courbe ROC en cours...")
        
        # Si le modèle est un pipeline, récupérer le classifieur
        if hasattr(model, 'named_steps') and 'clf' in model.named_steps:
            model_to_use = model
        else:
            model_to_use = model

        # Prédictions des probabilités
        y_proba = model_to_use.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)

        # Tracer
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="darkorange")
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel("Taux de faux positifs")
        plt.ylabel("Taux de vrais positifs")
        plt.title("Courbe ROC")
        plt.legend(loc="lower right")

        # Sauvegarde
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/roc_curve.png", bbox_inches='tight', dpi=300)
        plt.close()
        st.success("Courbe ROC générée avec succès.")
        st.image("images/roc_curve.png")

    except Exception as e:
        st.error(f"Erreur dans la génération de la courbe ROC : {str(e)}")
        raise


def shap_analysis(model, X, max_display=20):
    """
    Effectue l'analyse SHAP et sauvegarde/génère les visualisations
    
    Args:
        model: Le modèle sklearn/pipeline entraîné
        X: DataFrame ou ndarray contenant les features
        max_display: Nombre max de features à afficher
    """
    import numpy as np
    import pandas as pd

    try:
        st.info("Calcul des valeurs SHAP en cours...")

        # Si X est un ndarray, convertir en DataFrame
        if isinstance(X, np.ndarray):
            # Si le modèle est un pipeline avec un préprocesseur, on peut essayer de récupérer les noms de colonnes
            feature_names = None
            if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
                preproc = model.named_steps["preprocessor"]
                if hasattr(preproc, "get_feature_names_out"):
                    feature_names = preproc.get_feature_names_out()
            X_df = pd.DataFrame(X, columns=feature_names) if feature_names is not None else pd.DataFrame(X)
        else:
            X_df = X

        # Vérification du type de modèle
        if hasattr(model, 'named_steps') and 'clf' in model.named_steps:
            model_to_explain = model.named_steps['clf']
        else:
            model_to_explain = model

        # Création de l'explainer adapté au modèle
        if str(type(model_to_explain)).endswith("LogisticRegression'>"):
            explainer = shap.LinearExplainer(model_to_explain, X_df)
        else:
            explainer = shap.Explainer(model_to_explain, X_df)

        # Échantillonnage pour vitesse
        sample = X_df.sample(min(100, len(X_df)), random_state=42)
        shap_values = explainer(sample)

        # Visualisation
        st.success("Génération des graphiques SHAP")
        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
        plt.tight_layout()

        # Sauvegarde
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/shap_plot.png", bbox_inches='tight', dpi=300)
        st.image("images/shap_plot.png")

    except Exception as e:
        st.error(f"Erreur dans l'analyse SHAP : {str(e)}")
        raise