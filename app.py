'''
URL Format 
http://localhost:8000
'''

from flask import Flask, render_template,request
import pickle
import os
import pandas as pd

model_path = os.path.abspath("../artifacts/best_model/best_model")
model_path_file_name = os.path.join(model_path,"model.pkl")
model=pickle.load(open(model_path_file_name, 'rb'))
app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def my_webpage():
    if request.method == "POST":
        try:
            if request.form:
                data_request = dict(request.form)
                print(data_request)
                num1 = float(data_request["First_Number"])
                input_data = pd.DataFrame([[num1]], columns=["area"])
                response = model.predict(input_data)[0][0]
                return render_template("index1.html", response=response)
        except Exception as e:
            return f"Something Went Wrong. Try After Sometime !! {e}"
    else:
        return render_template("index1.html")
    
if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)