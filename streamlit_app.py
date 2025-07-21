# streamlit_app.py

# ================================================
# CONTEXTE DU PROJET : Détection de souscripteurs
# ================================================
# Ce projet vise à prédire si un client acceptera de souscrire à un dépôt à terme bancaire,
# à partir de données collectées lors de campagnes de téléprospection.
# Il repose sur un modèle supervisé entraîné sur un jeu de données de type marketing.
# Le modèle prédit la probabilité de souscription, et des visualisations sont proposées
# pour évaluer les performances du modèle (ROC, SHAP, matrice de confusion...).

# Membres de l'équipe projet :
# Alessi Clotaire
# Angueko Guy-Martial
# Foussard Cédric
# Ordonneau Louis-Paul

# ================================================
# Import des bibliothèques
# ================================================
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from model.model_utils import plot_roc, shap_analysis
from preprocessing.data_preprocessing import preprocess_data

from streamlit_dynamic_filters import DynamicFilters

# ================================================
# Configuration de la page Streamlit
# ================================================
st.set_page_config(
    page_title="Prédiction Souscription Dépôt",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================
# Chargement du modèle sauvegardé
# ================================================
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("models/best_model.pkl")

# ================================================
# Fonction principale de l'application
# ================================================
def main():
    st.title("Prédiction de souscription à un dépôt à terme")

    # === Introduction ===
    with st.expander("**CONTEXTE DU PROJET**", expanded=True):
        st.markdown("""
        Ce projet de Data Science vise à prédire la propension d'un client à souscrire à un dépôt à terme, 
        à partir de ses caractéristiques socio-économiques et de l'historique de la campagne marketing.

        **Cette interface permet de :**
        1. Visualiser les performances du modèle (courbe ROC, graphique SHAP, rapport de classification...).
        2. Explorer les variables influentes via SHAP.
        3. Exporter les prédictions et les données enrichies.
        """)

    # === Section Explications pour vulgarisation ===
    with st.expander("**EXPLICATIONS (POUR TOUS PUBLICS)**", expanded=True):
        st.markdown("""
        **Comment lire les graphiques ?**

        - **Courbe ROC** : Permet d'évaluer la capacité du modèle à distinguer les souscripteurs des non-souscripteurs. Plus la courbe est proche du coin supérieur gauche, meilleur est le modèle.
        - **Graphique SHAP** : Explique l'influence de chaque variable sur la prédiction du modèle, pour chaque client.
        """)

    # === Upload fichier ===
    st.sidebar.header("Paramètres")
    uploaded_file = st.sidebar.file_uploader("Charger un fichier CSV", type=["csv"])

    # Initialisation des variables d'état pour l'affichage unique des graphes et filtres
    for var in ["roc_displayed", "shap_displayed", "show_dynamic_filters"]:
        if var not in st.session_state:
            st.session_state[var] = False if var != "show_dynamic_filters" else True

    if uploaded_file is not None:
        with st.spinner("Lecture du fichier..."):
            try:
                # CORRECTION: Lecture directe sans fichier temporaire pour éviter les erreurs
                try:
                    df = pd.read_csv(uploaded_file, sep=",")
                except:
                    df = pd.read_csv(uploaded_file, sep=";")
                
                # CORRECTION: Vérification que les données ont été chargées correctement
                if df is None or df.empty:
                    st.error("Le fichier semble être vide ou n'a pas pu être lu correctement")
                    return
                
                df.columns = df.columns.str.strip()
                st.success("Fichier chargé avec succès!")

                st.subheader("Aperçu du dataset chargé")
                st.write(f"Dimensions: {df.shape[0]} lignes, {df.shape[1]} colonnes")
                st.dataframe(df.head())

                # CORRECTION: Vérification de l'existence des colonnes nécessaires
                required_cols = [
                            "pdays", "emp.var.rate", "cons.conf.idx", "psuccess",
                            "job_blue-collar", "job_entrepreneur", "job_housemaid", "job_management", "job_retired",
                            "job_self-employed", "job_services", "job_student", "job_technician", "job_unemployed",
                            "marital_married", "marital_single",
                            "education_basic.6y", "education_basic.9y", "education_high.school", "education_illiterate",
                            "education_professional.course", "education_university.degree",
                            "default_unknown", "default_yes",
                            "housing_yes", "loan_yes",
                            "poutcome_nonexistent", "poutcome_success",
                            "age_categ_Jeunes", "age_categ_Personnes agées"
                        ]
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.warning(f"Colonnes manquantes détectées: {missing_cols}")

                has_y = "y" in df.columns
                
                # CORRECTION: Gestion d'erreur pour le preprocessing
                try:
                    X, y, preprocessor = preprocess_data(df)
                    if X is None or len(X) == 0:
                        st.error("Erreur lors du preprocessing des données")
                        return
                except Exception as e:
                    st.error(f"Erreur lors du preprocessing: {e}")
                    return
                
                # CORRECTION: Gestion d'erreur pour le chargement du modèle
                try:
                    model = load_model()
                    if model is None:
                        st.error("Impossible de charger le modèle")
                        return
                except Exception as e:
                    st.error(f"Erreur lors du chargement du modèle: {e}")
                    return
                
                # CORRECTION: Vérification des prédictions
                try:
                    pred_proba = model.predict_proba(X)[:, 1]
                    predictions = (pred_proba > 0.5).astype(int)
                    
                    df["Probabilité"] = pred_proba
                    df["Prédit"] = predictions
                    df_positive = df[df["Prédit"] == 1].copy()
                    df_filtered = df.copy()
                except Exception as e:
                    st.error(f"Erreur lors des prédictions: {e}")
                    return

                # ==========================
                # Filtres globaux (âge, probabilité)
                # ==========================
                st.sidebar.markdown("### Filtres globaux")
                # Filtre sur l'âge (si présent)
                if "age" in df.columns:
                    age_min, age_max = int(df["age"].min()), int(df["age"].max())
                    age_range = st.sidebar.slider("Tranche d'âge", min_value=age_min, max_value=age_max, value=(age_min, age_max))
                    df_filtered = df_filtered[df_filtered["age"].between(age_range[0], age_range[1])]
                # Filtre sur la probabilité prédite
                prob_min, prob_max = float(df["Probabilité"].min()), float(df["Probabilité"].max())
                prob_range = st.sidebar.slider("Probabilité prédite", min_value=prob_min, max_value=prob_max, value=(prob_min, prob_max))
                df_filtered = df_filtered[df_filtered["Probabilité"].between(prob_range[0], prob_range[1])]

                # ==========================
                # Menu de visualisation
                # ==========================
                st.sidebar.markdown("---")
                st.sidebar.subheader("Visualisations")
                graph_option = st.sidebar.radio("Choisissez une vue :", [
                    "Aperçu des données",
                    "Résumé statistique global",
                    "Résultats de prédiction",
                    "Rapport de classification",
                    "Courbe ROC",
                    "Analyse SHAP",
                    # "Matrice de confusion",
                    "Distribution de la variable cible",
                    "Boxplot par variable",
                    "Tableau souscripteurs filtrés"
                ])

                # ==========================
                # Affichage des visualisations
                # ==========================
                if graph_option == "Aperçu des données":
                    st.subheader("Aperçu des données filtrées")
                    st.dataframe(df_filtered.head(20))

                elif graph_option == "Résumé statistique global":
                    st.subheader("Statistiques globales sur les variables numériques")
                    num_desc = df_filtered.describe().T
                    num_desc["median"] = df_filtered.median(numeric_only=True)
                    st.dataframe(num_desc[["min", "median", "mean", "max", "std"]])

                elif graph_option == "Résultats de prédiction":
                    st.subheader("Résultats de prédiction (filtrés)")
                    st.dataframe(df_filtered[["Probabilité", "Prédit"]].head(20))
                    st.download_button(
                        label="Télécharger les résultats",
                        data=df_filtered[["Probabilité", "Prédit"]].to_csv(index=False).encode("utf-8"),
                        file_name="resultats_predictions.csv",
                        mime="text/csv"
                    )

                elif graph_option == "Courbe ROC":
                    if has_y and not st.session_state["roc_displayed"]:
                        try:
                            # CORRECTION: Vérification de l'équilibre des classes pour diagnostiquer la courbe ROC diagonale
                            st.write(f"Distribution des classes - Classe 0: {sum(y==0)}, Classe 1: {sum(y==1)}")
                            st.write(f"Proportion classe positive: {sum(y==1)/len(y):.3f}")
                            
                            # CORRECTION: Vérification des prédictions du modèle
                            unique_preds = np.unique(predictions)
                            st.write(f"Prédictions uniques: {unique_preds}")
                            st.write(f"Proportion prédite positive: {sum(predictions==1)/len(predictions):.3f}")
                            
                            # MODIFICATION: Création et affichage sécurisé de la figure
                            # fig, ax = plt.subplots(figsize=(10, 6))
                            # plot_roc(model, X, y, ax=ax)  # Modification: la fonction prend maintenant un ax en paramètre
                            # st.pyplot(fig)
                            # plt.close(fig)
                            
                            # NOUVELLE IMPLÉMENTATION SÉCURISÉE
                            fig = plt.figure(figsize=(10, 6))
                            plot_roc(model, X, y)  # Appel original sans modification
                            st.pyplot(fig)  # Passage explicite de la figure
                            plt.close(fig)
                            
                            st.session_state["roc_displayed"] = True
                        except Exception as e:
                            st.error(f"Erreur lors de la génération de la courbe ROC : {e}")
                            st.write("Détails de l'erreur:", str(e))
                    elif not has_y:
                        st.warning("La colonne 'y' est nécessaire pour tracer la courbe ROC.")
                    elif st.session_state["roc_displayed"]:
                        st.info("Courbe ROC déjà affichée. Changez de vue puis revenez pour réinitialiser.")

                elif graph_option == "Analyse SHAP":
                    if not st.session_state["shap_displayed"]:
                        try:
                            if preprocessor is None:
                                st.error("Preprocessor manquant")
                                return

                            feature_names = preprocessor.get_feature_names_out()
                            sample_size = min(1000, len(X))
                            X_sample = X[:sample_size]
                            X_shap = pd.DataFrame(X_sample, columns=feature_names)

                            # Appel modifié
                            fig = shap_analysis(model, X_shap)
                            
                            if fig:
                                st.pyplot(fig, clear_figure=True)  # clear_figure efface la mémoire
                                plt.close(fig)  # Fermeture explicite
                                st.session_state["shap_displayed"] = True

                        except Exception as e:
                            st.error(f"Erreur : {str(e)}")
                    else:
                        st.info("Analyse déjà affichée")

                elif graph_option == "Matrice de confusion":
                    if has_y:
                        cm = confusion_matrix(y, predictions)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                        ax.set_title("Matrice de confusion")
                        ax.set_xlabel("Prédit")
                        ax.set_ylabel("Réel")
                        st.pyplot(fig)
                        
                        # CORRECTION: Ajout d'informations détaillées sur la matrice
                        st.write(f"Vrais Positifs: {cm[1,1]}, Faux Positifs: {cm[0,1]}")
                        st.write(f"Vrais Négatifs: {cm[0,0]}, Faux Négatifs: {cm[1,0]}")
                        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
                        st.write(f"Précision globale: {accuracy:.3f}")
                    else:
                        st.warning("La matrice de confusion nécessite la colonne 'y'.")

                elif graph_option == "Rapport de classification":
                    if has_y:
                        report = classification_report(y, predictions, output_dict=True)
                        st.dataframe(pd.DataFrame(report).transpose())
                    else:
                        st.warning("Le rapport de classification nécessite la colonne 'y'.")

                elif graph_option == "Distribution de la variable cible":
                    if has_y:
                        st.subheader("Distribution de la variable cible (y)")
                        fig = px.histogram(df_filtered, x="y", color="y", title="Distribution de la variable cible")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Colonne 'y' absente.")

                elif graph_option == "Boxplot par variable":
                    num_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
                    if len(num_cols) >= 1:
                        var = st.selectbox("Variable numérique à visualiser", num_cols)
                        fig = px.box(df_filtered, y=var, points="all", title=f"Boxplot de {var}")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Pas de colonnes numériques disponibles.")

                elif graph_option == "Tableau souscripteurs filtrés":
                    st.subheader("Souscripteurs filtrés")
                    if not df_positive.empty:
                        # Bouton pour afficher/masquer les filtres dynamiques
                        if st.sidebar.button(
                            "Masquer les filtres dynamiques" if st.session_state["show_dynamic_filters"] else "Afficher les filtres dynamiques"
                        ):
                            st.session_state["show_dynamic_filters"] = not st.session_state["show_dynamic_filters"]

                        # Liste des features adaptée aux données réelles
                        dynamic_filter_features = [
                            "emp.var.rate", "pdays", "age_categ_Personnes agées", "psuccess", "default_unknown", "poutcome_success",
                            "housing_yes", "default_yes"
                        ]      
                        
                        # Liste des features adaptée aux données réelles
                        # dynamic_filter_features = [
                           # "pdays", "emp.var.rate", "cons.conf.idx", "psuccess",
                           # "job_blue-collar", "job_entrepreneur", "job_housemaid", "job_management", "job_retired",
                           # "job_self-employed", "job_services", "job_student", "job_technician", "job_unemployed",
                           # "marital_married", "marital_single",
                           # "education_basic.6y", "education_basic.9y", "education_high.school", "education_illiterate",
                           # "education_professional.course", "education_university.degree",
                           # "default_unknown", "default_yes",
                           # "housing_yes", "loan_yes",
                           # "poutcome_nonexistent", "poutcome_success",
                           # "age_categ_Jeunes", "age_categ_Personnes agées"
                        # ]
                        available_cols = [col for col in dynamic_filter_features if col in df_positive.columns]

                        # Initialisation par défaut avec les données complètes
                        df_to_display = df_positive.copy()
                        filters_applied = False  # Nouveau flag pour suivre l'état des filtres

                        if st.session_state["show_dynamic_filters"] and available_cols:
                            try:
                                dynamic_filters = DynamicFilters(df_positive, filters=available_cols)
                                dynamic_filters.display_filters(location="sidebar")
                                
                                # Récupération des données filtrées
                                filtered_data = dynamic_filters.display_df()
                                
                                if filtered_data is not None and not filtered_data.empty:
                                    df_to_display = filtered_data
                                    filters_applied = True
                                else:
                                    # CORRECTION: Message clair dans la sidebar
                                    st.sidebar.warning("Vos critères n'ont retourné aucun résultat.")
                                    st.info("Affichage des données originales (non filtrées)")
                                    
                            except Exception as e:
                                st.sidebar.error(f"Erreur de filtrage: {str(e)}")
                                st.info("Affichage des données originales suite à une erreur")

                        # Affichage conditionnel avec indicateur
                        if filters_applied:
                            st.success("Filtres appliqués avec succès")
                        else:
                            st.info("Affichage des données complètes (aucun filtre actif)")
                        
                        st.dataframe(df_to_display.head(20))
                        
                        # Bouton de téléchargement adapté
                        st.download_button(
                            label="Télécharger les données affichées",
                            data=df_to_display.to_csv(index=False).encode("utf-8"),
                            file_name="souscripteurs.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("Aucun souscripteur détecté dans les données.")

                # --- Réinitialisation des affichages ROC/SHAP si on change de visualisation ---
                # MODIFICATION: Remplacé par un bouton de réinitialisation explicite
                if 'reset_plots' not in st.session_state:
                    st.session_state.reset_plots = False

                if st.sidebar.button('Réinitialiser les graphiques ROC/SHAP'):
                    st.session_state.roc_displayed = False
                    st.session_state.shap_displayed = False
                    st.rerun()

                # --- Sidebar équipe ---
                st.sidebar.markdown("---")
                st.sidebar.markdown("### À propos de l'équipe")
                st.sidebar.info("""
- Alessi Clotaire [LinkedIn](https://www.linkedin.com/in/clotaire-alessi/) 
- Angueko Guy-Martial [LinkedIn](https://www.linkedin.com/in/guymartialangueko/) 
- Foussard Cédric [LinkedIn](https://www.linkedin.com/in/cedricfoussard/) 
- Ordonneau Loui-Paul [LinkedIn](https://www.linkedin.com/in/louispaulordonneau/) 
                """)

                # === Export PDF (Streamlit 1.33+ requis) ===
                # st.sidebar.markdown("---")
                # st.sidebar.subheader("Exporter un rapport PDF")
                # if hasattr(st, "report"):
                    # if st.sidebar.button("Exporter ce rapport en PDF"):
                        # st.report("Rapport d'analyse prédictive", body=st.session_state)
                # else:
                    # st.sidebar.info("La fonctionnalité d'export PDF nécessite Streamlit >= 1.33.")

            except Exception as e:
                st.error(f"Une erreur est survenue : {e}")
                # CORRECTION: Affichage de l'erreur complète pour debugging
                st.write("Détails complets de l'erreur:", str(e))
                import traceback
                st.write("Traceback:", traceback.format_exc())

    else:
        st.info("Veuillez charger un fichier CSV pour démarrer l'analyse.")

# ================================================
# Exécution
# ================================================
if __name__ == "__main__":
    main()