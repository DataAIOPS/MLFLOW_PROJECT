import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./../artifacts/data/cleaned_data/homeprices_clean.csv")
X = df[["area"]]
Y= df["price"]

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.5)

X_train.to_csv("./../artifacts/data/processed/X_train.csv")
X_test.to_csv("./../artifacts/data/processed/X_test.csv")
y_train.to_csv("./../artifacts/data/processed/y_train.csv")
y_test.to_csv("./../artifacts/data/processed/y_test.csv")
