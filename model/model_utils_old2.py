# model_utils.py

import shap
import matplotlib.pyplot as plt
import os
import streamlit as st
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd


def plot_roc(model, X, y):
    """
    Génère et affiche la courbe ROC pour un modèle donné.

    Args:
        model: Le modèle entraîné (pipeline ou estimator)
        X: Données de test (features)
        y: Données de test (target)
    """
    try:
        st.info("Génération de la courbe ROC en cours...")

        # Récupérer le classifieur si pipeline
        model_to_use = model.named_steps['clf'] if hasattr(model, 'named_steps') and 'clf' in model.named_steps else model

        if not hasattr(model_to_use, "predict_proba"):
            st.warning("Ce modèle ne supporte pas predict_proba. Courbe ROC non disponible.")
            return

        y_proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)

        # Tracé
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Taux de faux positifs")
        ax.set_ylabel("Taux de vrais positifs")
        ax.set_title("Courbe ROC")
        ax.legend(loc="lower right")

        # Affichage Streamlit
        st.pyplot(fig)

        # Sauvegarde (optionnelle)
        os.makedirs("images", exist_ok=True)
        fig.savefig("images/roc_curve.png", bbox_inches='tight', dpi=300)

    except Exception as e:
        st.error(f"Erreur lors de la génération de la courbe ROC : {str(e)}")
        raise


def shap_analysis(model, X, max_display=20):
    """
    Effectue l'analyse SHAP et affiche un beeswarm plot.

    Args:
        model: Modèle entraîné ou pipeline
        X: DataFrame ou ndarray contenant les features
        max_display: Nombre maximum de variables à afficher
    """
    try:
        st.info("Calcul des valeurs SHAP...")

        # Convertir X si nécessaire
        if isinstance(X, np.ndarray):
            feature_names = None
            if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
                preproc = model.named_steps["preprocessor"]
                if hasattr(preproc, "get_feature_names_out"):
                    feature_names = preproc.get_feature_names_out()
            X_df = pd.DataFrame(X, columns=feature_names) if feature_names is not None else pd.DataFrame(X)
        else:
            X_df = X

        # Extraire modèle interne
        model_to_explain = model.named_steps['clf'] if hasattr(model, 'named_steps') and 'clf' in model.named_steps else model

        # Sélection de l'explainer
        model_type = str(type(model_to_explain))
        if "LogisticRegression" in model_type:
            explainer = shap.LinearExplainer(model_to_explain, X_df)
        elif "Tree" in model_type or "Forest" in model_type or "Boosting" in model_type:
            explainer = shap.Explainer(model_to_explain, X_df)
        else:
            explainer = shap.Explainer(model_to_explain, X_df)

        sample = X_df.sample(min(100, len(X_df)), random_state=42)
        shap_values = explainer(sample)

        st.success("SHAP valeurs calculées. Génération du graphe...")

        # Tracé
        plt.figure(figsize=(8, 6))
        shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
        plt.tight_layout()

        # Sauvegarde et affichage
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/shap_plot.png", bbox_inches='tight', dpi=300)
        st.image("images/shap_plot.png")

    except Exception as e:
        st.error(f"Erreur dans l'analyse SHAP : {str(e)}")
        raise