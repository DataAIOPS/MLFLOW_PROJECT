import pandas as pd
import pickle
import os
import argparse
from sklearn.linear_model import LinearRegression

processed_data_path = os.path.abspath("./../artifacts/data/processed_data/")
model_path = os.path.abspath("./../artifacts/model/")

def model_building(processed_data_path,model_path):
    X_train_path = os.path.join(processed_data_path,"X_train.csv")
    y_train_path = os.path.join(processed_data_path,"y_train.csv")

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    print("[INFO] Model building is started")
    model = LinearRegression()
    model.fit(X_train,y_train)

    model_path_file_name=os.path.join(model_path,"my_model.pkl")
    pickle.dump(model,open(model_path_file_name,"wb"))
    print(f"[INFO] model is exporeted to {model_path_file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data_path", help="provide processed data path",default=processed_data_path)
    parser.add_argument("--model_path", help="provide model directory path",default=model_path)
    args = parser.parse_args()
    model_building(args.processed_data_path,args.model_path)






