import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split

cleaned_data_path = os.path.abspath("./../artifacts/data/cleaned_data/")
processed_data_path = os.path.abspath("./../artifacts/data/processed_data/")


def processed_data(cleaned_data_path,processed_data_path,Target):
    print(cleaned_data_path)
    cleaned_data = os.listdir(cleaned_data_path)[0]
    cleaned_data = os.path.join(cleaned_data_path,cleaned_data)
    df = pd.read_csv(cleaned_data)
    Y = df[[Target]]
    X = df.drop(columns=[Target])

    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.5)

    print("X =",X_test)
    print()
    print("Y =",y_test)

    X_train.to_csv(os.path.join(processed_data_path,"X_train.csv"),index=False)
    X_test.to_csv(os.path.join(processed_data_path,"X_test.csv"),index=False)
    y_train.to_csv(os.path.join(processed_data_path,"y_train.csv"),index=False)
    y_test.to_csv(os.path.join(processed_data_path,"y_test.csv"),index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned_data_path", help="provide clean data path",default=cleaned_data_path)
    parser.add_argument("--processed_data_path", help="provide processed data path",default=processed_data_path)
    parser.add_argument("--Target", help="provide Target Variable name",default=None)
    args = parser.parse_args()
    print(args)
    if args.Target != None:
        processed_data(args.cleaned_data_path,args.processed_data_path,args.Target)
    else:
        print(f"[ERROR] Something went wrong. Target variable cant be None {args.Target}")

