from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import pandas as pd

model=pickle.load(open("./../artifacts/model/model.pkl", 'rb'))

X_test = pd.read_csv("./../artifacts/data/processed/X_test.csv")
y_test = pd.read_csv("./../artifacts/data/processed/y_test.csv")

y_pred_test = model.predict(X_test)

r_score=r2_score(y_test,y_pred_test)
mean_squared_error=mean_squared_error(y_test,y_pred_test)
mean_absolute_error=mean_absolute_error(y_test,y_pred_test)

if r_score > 0.8:
    print(r_score)
    pickle.dump(model,open("./../artifacts/model_eval/model.pkl","wb"))

