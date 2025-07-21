import joblib

model = joblib.load("models/best_model.pkl")
print("Type du modèle :", type(model))

if hasattr(model, 'named_steps'):
    print("\nPipeline steps :")
    for name, step in model.named_steps.items():
        print(f"  - {name}: {type(step)}")
else:
    print("Le modèle n'est pas un Pipeline.")