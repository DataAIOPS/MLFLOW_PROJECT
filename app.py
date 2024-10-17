import os
import pickle
from flask import Flask, render_template,request

model_path = os.path.abspath("/app/artifacts/best_model/best_model")
model_path_file_name = os.path.join(model_path,"model.pkl")
model=pickle.load(open(model_path_file_name, 'rb'))

app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def prediction():
    if request.method == "POST":
        data_dict =dict(request.form)
        area = float(data_dict["area"])
        prediction = model.predict([[area]])[0][0]
        return render_template("test.html",prediction=prediction)
    else:
        return render_template("test.html")

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)