from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
def load_data(path):
    import pandas as pd
    return pd.read_csv(path)
def clean_data(df):
    df = df.drop(columns=["id"])
    X = df.drop(columns=["num"])
    y = df["num"]

    X = pd.get_dummies(X)
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test