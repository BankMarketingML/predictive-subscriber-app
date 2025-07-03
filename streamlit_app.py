import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import plotly.express as px

from model.model_utils import plot_roc, shap_analysis
from preprocessing.data_preprocessing import preprocess_data

# Configuration de la page
st.set_page_config(
    page_title="🔍 Prédiction Souscription Dépôt",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Liste attendue des colonnes (hors 'y')
REQUIRED_COLUMNS = [
    'age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
    'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome',
    'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'psucess'
]

@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("models/best_model.pkl")

def main():
    st.title("📊 Prédiction de souscription à un dépôt à terme")

    uploaded_file = st.file_uploader("📂 Chargez votre fichier CSV", type=["csv"])
    
    if uploaded_file is not None:
        with st.spinner("🔄 Lecture du fichier..."):
            os.makedirs("data", exist_ok=True)
            temp_path = "data/temp.csv"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                # Lecture intelligente
                try:
                    df = pd.read_csv(temp_path)
                except:
                    df = pd.read_csv(temp_path, sep=";")

                st.subheader("👁️ Aperçu des données")
                st.dataframe(df.head())

                # Vérification des colonnes
                missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
                if missing_cols:
                    st.error(f"❌ Le fichier est invalide. Il manque les colonnes suivantes : {missing_cols}")
                    return

                has_y = 'y' in df.columns

                # Prétraitement
                X, y, preprocessor = preprocess_data(df)
                model = load_model()

                # Prédictions
                pred_proba = model.predict_proba(X)[:, 1]
                predictions = (pred_proba > 0.5).astype(int)

                df["Probabilité"] = pred_proba
                df["Prédit"] = predictions

                # === Affichage des prédictions principales
                st.subheader("✅ Résultats de prédiction")
                results_df = pd.DataFrame({
                    "Probabilité": pred_proba,
                    "Prédiction": predictions
                })
                st.dataframe(results_df.head(10))

                # === Téléchargement principal
                csv_results = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Télécharger les résultats de prédiction",
                    data=csv_results,
                    file_name="predictions_souscription.csv",
                    mime="text/csv"
                )

                # === Courbe ROC
                if has_y:
                    if st.button("📉 Générer la courbe ROC"):
                        try:
                            plot_roc(model, X, y)
                            st.image("images/roc_curve.png", caption="Courbe ROC")
                        except Exception as e:
                            st.error(f"Erreur lors de la génération de la courbe ROC : {e}")
                else:
                    st.warning("ℹ️ La courbe ROC nécessite la présence de la colonne cible 'y'.")

                # === SHAP
                if st.button("🧠 Lancer l’analyse SHAP"):
                    try:
                        if isinstance(X, np.ndarray):
                            feature_names = (
                                preprocessor.get_feature_names_out()
                                if hasattr(preprocessor, "get_feature_names_out")
                                else None
                            )
                            X_shap = pd.DataFrame(X, columns=feature_names) if feature_names is not None else pd.DataFrame(X)
                        else:
                            X_shap = X
                        shap_analysis(model, X_shap)
                        st.image("images/shap_plot.png", caption="Graphique SHAP")
                    except Exception as e:
                        st.error(f"Erreur SHAP : {str(e)}")

                # === Souscripteurs détectés
                df_positive = df[df["Prédit"] == 1]
                if not df_positive.empty:
                    st.subheader("🎯 Souscripteurs potentiels détectés")
                    st.dataframe(df_positive.head(10))

                    # Filtres
                    with st.expander("🎛️ Affiner les résultats"):
                        jobs = df_positive["job"].dropna().unique().tolist()
                        selected_job = st.multiselect("Métier", options=jobs, default=jobs)

                        age_min, age_max = int(df_positive["age"].min()), int(df_positive["age"].max())
                        age_range = st.slider("Tranche d'âge", min_value=age_min, max_value=age_max, value=(age_min, age_max))

                        duration_min, duration_max = int(df_positive["duration"].min()), int(df_positive["duration"].max())
                        duration_range = st.slider("Durée de l'appel", min_value=duration_min, max_value=duration_max, value=(duration_min, duration_max))

                        df_positive_filtered = df_positive[
                            (df_positive["job"].isin(selected_job)) &
                            (df_positive["age"].between(age_range[0], age_range[1])) &
                            (df_positive["duration"].between(duration_range[0], duration_range[1]))
                        ]
                        if not df_positive_filtered.empty:
                            st.subheader("📊 Analyse visuelle des souscripteurs filtrés")

                            chart_type = st.radio("Sélectionner un graphique :", ["Par tranche d’âge", "Par métier"])

                            if chart_type == "Par tranche d’âge":
                                df_positive_filtered["tranche_age"] = pd.cut(
                                    df_positive_filtered["age"],
                                    bins=[18, 25, 35, 45, 55, 65, 100],
                                    labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
                                    right=False
                                )
                                age_fig = px.histogram(df_positive_filtered, x="tranche_age", color_discrete_sequence=["#636EFA"])
                                age_fig.update_layout(title="Nombre de souscripteurs par tranche d’âge", xaxis_title="Tranche d’âge", yaxis_title="Nombre")
                                st.plotly_chart(age_fig, use_container_width=True)

                            elif chart_type == "Par métier":
                                job_fig = px.bar(
                                    df_positive_filtered["job"].value_counts().reset_index(),
                                    x="index", y="job",
                                    labels={"index": "Métier", "job": "Nombre de souscripteurs"},
                                    color_discrete_sequence=["#EF553B"]
                                )
                                job_fig.update_layout(title="Nombre de souscripteurs par métier")
                                st.plotly_chart(job_fig, use_container_width=True)

                    # 📥 Boutons de téléchargement (complet + filtré)
                    st.subheader("📤 Télécharger les résultats")

                    col1, col2 = st.columns(2)

                    with col1:
                        csv_full = df_positive.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Télécharger tous les souscripteurs détectés",
                            data=csv_full,
                            file_name="souscripteurs_complets.csv",
                            mime="text/csv"
                        )

                    with col2:
                        csv_filtered = df_positive_filtered.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Télécharger les souscripteurs filtrés",
                            data=csv_filtered,
                            file_name="souscripteurs_filtres.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"❌ Une erreur est survenue lors du traitement : {str(e)}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

if __name__ == "__main__":
    main()