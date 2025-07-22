from src.preprocessing import load_data, clean_data
from src.model import train_model
from src.evaluation import evaluate_model

# Cargar y preparar los datos
df = load_data('Proyecto_Heart_Disease/Data/heart_disease_uci.csv')
X_train, X_test, y_train, y_test = clean_data(df)

# Entrenar modelo
model = train_model(X_train, y_train)

# Evaluar
evaluate_model(model, X_test, y_test)

