from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import pandas as pd
import argparse
import mlflow

processed_data_path = os.path.abspath("./../artifacts/data/processed_data/")
model_path = os.path.abspath("./../artifacts/model/")
eval_model_path = os.path.abspath("./../artifacts/model_eval")

def eval_model(processed_data_path,model_path,eval_model_path):
    print("#################Model Evaluation Started################")
    model_path_file_name = os.path.join(model_path,"my_model.pkl")
    model=pickle.load(open(model_path_file_name, 'rb'))

    X_test_path = os.path.join(processed_data_path,"X_test.csv")
    y_test_path = os.path.join(processed_data_path,"y_test.csv")

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    y_pred_test = model.predict(X_test)

    r_score=r2_score(y_test,y_pred_test)
    MSE = mean_squared_error(y_test,y_pred_test)
    MAE = mean_absolute_error(y_test,y_pred_test)
    mlflow.log_metrics({"R_square":r_score,
                        "MSE":MSE,
                        "MAE":MAE})
    if r_score > 0.8:
        print(r_score)
        eval_model_path_file = os.path.join(eval_model_path,"eval_model.pkl")
        pickle.dump(model,open(eval_model_path_file,"wb"))
        mlflow.sklearn.log_model(model,"best_model")
    else:
        print(f"[WARNING] model doesnt pass evalution criteria r_score = {r_score}")
    print("#################Model Evaluation Finished ################")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data_path", help="provide processed data path",default=processed_data_path)
    parser.add_argument("--model_path", help="provide model directory path",default=model_path)
    parser.add_argument("--eval_model_path", help="provide eval_model_path directory path",default=eval_model_path)
    args = parser.parse_args()
    eval_model(args.processed_data_path,args.model_path,args.eval_model_path)

