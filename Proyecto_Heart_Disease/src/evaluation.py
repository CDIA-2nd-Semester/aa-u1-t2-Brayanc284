from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Exactitud: {acc:.2f}")
    print("\n📋 Reporte de Clasificación:\n", classification_report(y_test, y_pred))
    print("\n🧾 Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
