import pandas as pd
import argparse
import os
import mlflow
raw_data_path = os.path.abspath("./../artifacts/data/raw_data/")
clean_data_path = os.path.abspath("./../artifacts/data/cleaned_data/")
raw_data_file = "homeprices.csv"


def generate_file_name(file_name):
    revised_file_name = file_name.split(".csv")
    revised_file_name.insert(-1,"clean.csv")
    revised_file_name.pop()
    revised_file_name = "_".join(revised_file_name)
    return revised_file_name


def data_cleaning(raw_data_path,clean_data_path,raw_data_file):
    print("################DATA CLEANING STARTED##################")
    raw_data = os.path.join(raw_data_path,raw_data_file)
    df = pd.read_csv(raw_data)
    
    clean_data_file = generate_file_name(raw_data_file)
    clean_data = os.path.join(clean_data_path,clean_data_file)
    df.to_csv(clean_data,index=False)
    mlflow.log_param("clean_data_path",clean_data)
    print(f"[INFO] exported clean data at {clean_data}")
    print("################DATA CLEANING FINISHED##################")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", help="provide raw data path",default=raw_data_path)
    parser.add_argument("--clean_data_path", help="provide clean data path",default=clean_data_path)
    parser.add_argument("--raw_data_file", help="provide raw data file name",default=raw_data_file)
    args = parser.parse_args()
    data_cleaning(args.raw_data_path,args.clean_data_path,args.raw_data_file)



