import streamlit as st
import pandas as pd
import joblib
import os
from model.model_utils import plot_roc, shap_analysis
from preprocessing.data_preprocessing import preprocess_data

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Souscription Dépôt",
    layout="wide",
    initial_sidebar_state="auto"
)

@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("models/best_model.pkl")

def main():
    st.title("🔍 Prédiction de la souscription à un dépôt à terme")

    uploaded_file = st.file_uploader("Chargez votre fichier CSV", type=["csv"])
    if uploaded_file is not None:
        with st.spinner("Traitement en cours..."):
            temp_path = "data/temp.csv"
            os.makedirs("data", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                # Tentative auto du séparateur
                try:
                    df = pd.read_csv(temp_path)
                except Exception:
                    df = pd.read_csv(temp_path, sep=";")

                st.subheader("Aperçu des données chargées")
                st.dataframe(df.head())

                # Prétraitement
                X, y, preprocessor = preprocess_data(df)
                model = load_model()

                pred_proba = model.predict_proba(X)[:, 1]
                predictions = (pred_proba > 0.5).astype(int)

                # Affichage des résultats
                st.subheader("✅ Prédictions")
                results_df = pd.DataFrame({
                    "Probabilité": pred_proba,
                    "Prédit": predictions
                })
                st.dataframe(results_df.head(10))

                # Courbe ROC (si y connu)
                if y is not None and hasattr(model, "predict_proba"):
                    try:
                        st.subheader("📈 Courbe ROC")
                        plot_roc(model, X, y)
                        st.image("images/roc_curve.png")
                    except Exception as e:
                        st.error(f"Erreur génération ROC: {str(e)}")
                else:
                    st.info("La courbe ROC n'est disponible que si la colonne cible 'y' est présente dans le fichier.")

                # Analyse SHAP
                try:
                    st.subheader("📊 Analyse SHAP")
                    # Conversion X en DataFrame si nécessaire pour SHAP
                    import numpy as np
                    if isinstance(X, np.ndarray):
                        # Si possible, récupérer les noms de colonnes depuis le préprocesseur
                        feature_names = None
                        if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
                            feature_names = preprocessor.get_feature_names_out()
                        X_shap = pd.DataFrame(X, columns=feature_names) if feature_names is not None else pd.DataFrame(X)
                    else:
                        X_shap = X
                    shap_analysis(model, X_shap)
                    st.image("images/shap_plot.png")
                except Exception as e:
                    st.error(f"Erreur analyse SHAP: {str(e)}")

            except Exception as e:
                st.error(f"Une erreur critique est survenue: {str(e)}")

            finally:
                # Nettoyage du fichier temporaire
                if os.path.exists(temp_path):
                    os.remove(temp_path)

if __name__ == "__main__":
    main()