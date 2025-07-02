# generate_images.py
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import shap
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline

# 1. Configuration initiale
os.makedirs("images", exist_ok=True)

# 2. Charger le modèle
print("Chargement du modèle...")
model = joblib.load("models/best_model.pkl")

# 3. Analyse de la structure du modèle
print("\n=== Structure du modèle ===")
print("Type complet du modèle:", str(type(model)))
is_pipeline = isinstance(model, (Pipeline, ImbPipeline))
print("Est un pipeline?", is_pipeline)

if hasattr(model, 'named_steps'):
    print("\nÉtapes du pipeline:")
    for step_name, step in model.named_steps.items():
        print(f"- {step_name}: {type(step)}")
else:
    print("\nLe modèle n'est pas un pipeline ou n'a pas d'étapes nommées")

# 4. Charger les données
print("\nChargement des données...")
df = pd.read_csv("data/bank_data.csv", sep=';')
print("Colonnes disponibles:", df.columns.tolist())

if 'y' not in df.columns:
    raise ValueError("Colonne cible 'y' manquante")

X = df.drop(columns=['y'])
y = df['y']

# 5. Reproduction du preprocessing original
def reproduce_preprocessing(X):
    """
    Reproduit le preprocessing utilisé lors de l'entraînement du modèle
    basé sur les noms de features attendues par le modèle
    """
    print("\nReproduction du preprocessing original...")
    X_processed = X.copy()
    
    # 1. Création de catégories d'âge
    if 'age' in X_processed.columns:
        def categorize_age(age):
            if age <= 30:
                return 'Jeunes'
            elif age >= 60:
                return 'Personnes agées'
            else:
                return 'Adultes'
        
        X_processed['age_categ'] = X_processed['age'].apply(categorize_age)
        # Encodage one-hot pour age_categ
        age_dummies = pd.get_dummies(X_processed['age_categ'], prefix='age_categ')
        X_processed = pd.concat([X_processed, age_dummies], axis=1)
        X_processed.drop(['age', 'age_categ'], axis=1, inplace=True)
    
    # 2. Création de la variable psuccess
    if 'poutcome' in X_processed.columns:
        X_processed['psuccess'] = (X_processed['poutcome'] == 'success').astype(int)
    
    # 3. Encodage des variables catégorielles restantes
    categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        X_processed = pd.get_dummies(X_processed, columns=categorical_cols, drop_first=False)
    
    # 4. Conversion de la cible y si elle existe encore
    if 'y' in X_processed.columns:
        X_processed['y'] = (X_processed['y'] == 'yes').astype(int)
    
    return X_processed

# 6. Transformation des données avec preprocessing reproduit
print("\nTransformation des données...")
try:
    X_processed = reproduce_preprocessing(X)
    print("✅ Preprocessing reproduit avec succès")
    print("Features après preprocessing:", X_processed.columns.tolist())
    
    # Vérification que toutes les features attendues sont présentes
    try:
        # Test de prédiction sur un petit échantillon pour vérifier la compatibilité
        test_sample = X_processed.head(1)
        _ = model.predict_proba(test_sample)
        print("✅ Compatibilité des features vérifiée")
        X_final = X_processed
    except Exception as e:
        print(f"⚠️ Ajustement nécessaire des features: {str(e)}")
        # Charger le modèle entraîné pour obtenir les noms de features
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
        else:
            # Essayer d'obtenir les features depuis le premier step du pipeline
            try:
                scaler = model.named_steps['scaler']
                if hasattr(scaler, 'feature_names_in_'):
                    expected_features = scaler.feature_names_in_
                else:
                    raise AttributeError("Impossible d'obtenir les noms de features")
            except:
                print("❌ Impossible d'obtenir les noms de features attendues")
                raise
        
        print("Features attendues:", list(expected_features))
        
        # Ajuster les colonnes pour correspondre exactement
        missing_cols = set(expected_features) - set(X_processed.columns)
        extra_cols = set(X_processed.columns) - set(expected_features)
        
        if missing_cols:
            print(f"Colonnes manquantes: {missing_cols}")
            for col in missing_cols:
                X_processed[col] = 0  # Ajouter avec valeur par défaut
        
        if extra_cols:
            print(f"Colonnes supplémentaires à supprimer: {extra_cols}")
            X_processed = X_processed.drop(columns=list(extra_cols))
        
        # Réorganiser les colonnes dans l'ordre attendu
        X_final = X_processed[expected_features]
        print("✅ Features ajustées pour correspondre au modèle")

except Exception as e:
    print(f"❌ Erreur critique de preprocessing: {str(e)}")
    raise

# 7. Génération de la courbe ROC (version ultra-robuste)
def generate_roc():
    try:
        print("\nGénération de la courbe ROC...")
        
        # Gestion de tous les types de modèles
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X_final)[:, 1]
        elif hasattr(model, 'decision_function'):
            probas = model.decision_function(X_final)
            probas = (probas - probas.min()) / (probas.max() - probas.min())  # Normalisation
        else:
            raise ValueError("Le modèle n'a ni predict_proba ni decision_function")
        
        fpr, tpr, _ = roc_curve(y, probas)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig("images/roc_curve.png", bbox_inches='tight', dpi=300)
        plt.close()
        print("✅ ROC curve générée avec succès")
    except Exception as e:
        print(f"❌ Erreur ROC critique: {str(e)}")
        raise

# 8. Génération du plot SHAP (version ultra-robuste)
def generate_shap():
    try:
        print("\nGénération du SHAP plot...")
        
        # Identification du modèle à expliquer (le classifieur final)
        model_to_explain = model
        if hasattr(model, 'named_steps') and 'clf' in model.named_steps:
            # Pour expliquer avec SHAP, on a besoin du classifieur seul
            # On va créer une fonction wrapper
            classifier = model.named_steps['clf']
            
            def predict_fn(X):
                # X est déjà transformé par les étapes précédentes du pipeline
                return classifier.predict_proba(X)[:, 1]
        else:
            predict_fn = lambda X: model.predict_proba(X)[:, 1]
        
        # Échantillonnage pour performance
        sample_size = min(100, len(X_final))
        sample_indices = np.random.choice(len(X_final), sample_size, replace=False)
        X_sample = X_final.iloc[sample_indices]
        
        # Pour SHAP, on a besoin des données transformées jusqu'au classifieur
        if hasattr(model, 'named_steps'):
            # Appliquer toutes les transformations sauf le classifieur final
            X_transformed = X_sample.copy()
            for step_name, step in model.named_steps.items():
                if step_name != 'clf' and hasattr(step, 'transform'):
                    X_transformed = step.transform(X_transformed)
                elif step_name == 'clf':
                    break
            
            # Convertir en DataFrame si nécessaire
            if not isinstance(X_transformed, pd.DataFrame):
                X_transformed = pd.DataFrame(X_transformed)
        else:
            X_transformed = X_sample
        
        # Création de l'explainer adaptatif
        try:
            # Pour LogisticRegression, utiliser LinearExplainer
            if 'LogisticRegression' in str(type(model.named_steps.get('clf', model))):
                explainer = shap.LinearExplainer(model.named_steps['clf'], X_transformed)
                shap_values = explainer.shap_values(X_transformed)
            else:
                # Utiliser KernelExplainer comme solution universelle
                explainer = shap.KernelExplainer(predict_fn, X_transformed.sample(min(50, len(X_transformed))))
                shap_values = explainer.shap_values(X_transformed)
        except Exception as shap_error:
            print(f"Erreur avec les explainers spécialisés: {shap_error}")
            print("Utilisation de KernelExplainer comme solution alternative")
            explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], X_sample.sample(min(20, len(X_sample))))
            shap_values = explainer.shap_values(X_sample)
            X_transformed = X_sample  # Pour les noms de features
        
        # Visualisation
        plt.figure(figsize=(12, 8))
        if hasattr(shap_values, 'values'):  # Nouveau format SHAP
            shap.summary_plot(shap_values.values, X_transformed, max_display=15, show=False)
        else:  # Ancien format
            shap.summary_plot(shap_values, X_transformed, max_display=15, show=False)
        
        plt.tight_layout()
        plt.savefig("images/shap_plot.png", bbox_inches='tight', dpi=300)
        plt.close()
        print("✅ SHAP plot généré avec succès")
    except Exception as e:
        print(f"❌ Erreur SHAP: {str(e)}")
        print("Génération d'un graphique d'importance des features alternatif...")
        
        # Solution de fallback : importance des coefficients pour LogisticRegression
        try:
            if hasattr(model, 'named_steps') and 'clf' in model.named_steps:
                clf = model.named_steps['clf']
                if hasattr(clf, 'coef_'):
                    feature_importance = np.abs(clf.coef_[0])
                    feature_names = X_final.columns if hasattr(X_final, 'columns') else [f'Feature_{i}' for i in range(len(feature_importance))]
                    
                    # Top 15 features les plus importantes
                    top_indices = np.argsort(feature_importance)[-15:]
                    
                    plt.figure(figsize=(10, 6))
                    plt.barh(range(len(top_indices)), feature_importance[top_indices])
                    plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
                    plt.xlabel('Importance (Coefficient absolu)')
                    plt.title('Importance des Features (Coefficients)')
                    plt.tight_layout()
                    plt.savefig("images/feature_importance.png", bbox_inches='tight', dpi=300)
                    plt.close()
                    print("✅ Graphique d'importance alternatif généré")
        except Exception as fallback_error:
            print(f"❌ Erreur avec le fallback: {fallback_error}")

# 9. Exécution principale
if __name__ == "__main__":
    print("\n=== Début de la génération ===")
    generate_roc()
    generate_shap()
    
    print("\n=== Vérification ===")
    if os.path.exists("images"):
        print("Fichiers générés dans images/:")
        print(os.listdir("images"))
    print("\nOpération terminée avec succès!")