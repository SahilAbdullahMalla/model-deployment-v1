from flask import Flask, render_template, request
import pickle 
import os

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(__file__)

# Build full paths to the model files
cv_path = os.path.join(BASE_DIR, "models", "cv.pkl")
clf_path = os.path.join(BASE_DIR, "models", "clf.pkl")

# Load the models
with open(cv_path, "rb") as f:
    tokenizer = pickle.load(f)

with open(clf_path, "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
   
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    
    email=request.form.get("content")
    tokenized_email=tokenizer.transform([email])
    predictions=model.predict(tokenized_email)
    predictions=1 if predictions==1 else -1
    return render_template("index.html",predictions=predictions,email=email)

if __name__ == "__main__":
    app.run(port=7000, debug=True)

