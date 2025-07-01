# Predictive Subscriber App

Ce projet vise à prédire si un client va souscrire à un dépôt à terme, à partir d’un jeu de données marketing bancaire.

## 📁 Structure du projet

predictive_subscriber_app/
├── app/
│ └── streamlit_app.py
├── data/
│ └── bank_data.csv
├── model/
│ └── best_model.pkl
├── models/
│ └── model_training.py
├── preprocessing/
│ └── data_preprocessing.py
├── images/
│ └── shap_plot.png
├── README.md


## 🚀 Utilisation

1. Installe les dépendances :
pip install -r requirements.txt

2. Lance l'application Streamlit :
streamlit run app/streamlit_app.py


## 📊 Interprétabilité

Le projet utilise SHAP pour expliquer les prédictions du modèle.

## 🛠️ Librairies principales

- `scikit-learn`
- `imblearn`
- `shap`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `streamlit`

## 📁 Auteurs

- Alessi Clotaire
- Angueko Guy-Martial
- Foussard Cédric
- Ordonneau Louis-Paul