import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

X_train = pd.read_csv("./../artifacts/data/processed/X_train.csv")
y_train = pd.read_csv("./../artifacts/data/processed/y_train.csv")

model = LinearRegression()
model.fit(X_train,y_train)

pickle.dump(model,open("./../artifacts/model/model.pkl","wb"))


