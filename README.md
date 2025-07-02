# 💡 Prédiction de Souscription à un Dépôt à Terme

Cette application Streamlit permet de prédire la probabilité qu’un client souscrive à un dépôt à terme, à partir de données clients historiques. Elle s'appuie sur un modèle de machine learning optimisé et interprété à l’aide de SHAP et d’indicateurs classiques (ROC, précision, rappel...).

## 🚀 Fonctionnalités

- 📊 Chargement de données CSV
- 🧹 Prétraitement automatisé (encodage, standardisation, gestion des valeurs manquantes)
- 🤖 Prédiction via un modèle pré-entraîné
- 📈 Affichage de la courbe ROC
- 🔍 Analyse des variables explicatives via SHAP
- ✅ Visualisation des résultats

## 📂 Arborescence

predictive-subscriber-app/
│
├── app/
│ └── streamlit_app.py # Application Streamlit principale
│
├── preprocessing/
│ └── data_preprocessing.py # Nettoyage et transformation des données
│
├── model/
│ └── model_utils.py # Fonctions de visualisation et SHAP
│
├── models/
│ └── best_model.pkl # Modèle ML sauvegardé
│
├── images/
│ ├── roc_curve.png # Courbe ROC (générée)
│ └── shap_plot.png # Analyse SHAP (générée)
│
├── requirements.txt # Dépendances Python
└── README.md # Ce fichier

markdown
Copier
Modifier

## 📌 Exemple de jeu de données attendu

Votre fichier CSV doit contenir les colonnes utilisées pour entraîner le modèle, comme par exemple : `age`, `job`, `balance`, `marital`, etc. La colonne cible `y` n'est pas requise pour la prédiction.

## ▶️ Lancer l’application en local

1. Cloner ce dépôt :
   ```bash
   git clone https://github.com/BankMarketingML/predictive-subscriber-app.git
   cd predictive-subscriber-app
Installer les dépendances :

bash
Copier
Modifier
pip install -r requirements.txt
Lancer Streamlit :

bash
Copier
Modifier
streamlit run app/streamlit_app.py
☁️ Déploiement sur Streamlit Cloud
Forker ce dépôt sur GitHub

Aller sur https://streamlit.io/cloud

Créer une nouvelle app avec ce dépôt

Spécifier le fichier principal : app/streamlit_app.py

🔒 Avertissement
Aucune donnée sensible ne doit être utilisée sans anonymisation. Ce projet est fourni à titre pédagogique.

👨‍💻 Auteurs
Projet réalisé, dans le cadre d’une formation en Data Science, par :
- Alessi Clotaire
- Angueko Guy-Martial
- Foussard Cédric
- Ordonneau Louis-Paul