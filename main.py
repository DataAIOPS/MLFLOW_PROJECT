import argparse
import mlflow

def main(Target):
    print(f"[INFO] MLOps Pipeline Triggerd for")
    with mlflow.start_run() as run:
        mlflow.run("./src", entry_point="Data_Cleaning.py",env_manager="local")
        mlflow.run("./src",entry_point="Data_Preprocessing.py",parameters={'Target':Target},env_manager="local")
        mlflow.run("./src", entry_point="Model_Building.py",env_manager="local")
        mlflow.run("./src","Model_Evaluation.py",env_manager="local")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--Target",type=str, default=None)
    args = parser.parse_args()
    main(args.Target)



        
