from flask import Flask, request, render_template, redirect, url_for,session
import repodownloader 
import ktrain
import pandas as pd
from ktrain import tabular
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import math
import os
import train_combined_model
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'


def apply_threshold(probabilities, threshold=0.28):
    return (probabilities >= threshold).astype(int)


def map_predictions(predictions, labels):
    # Map each binary prediction to its corresponding label or 'not_LA'
    mapped_predictions = []
    for prediction in predictions:
        # Get the labels for which the prediction is 1
        label_predictions = [label for pred, label in zip(prediction, labels) if pred == 1]
        
        # Join the labels into a string, or return 'not_LA' if no labels are present
        prediction_str = ','.join(label_predictions) if label_predictions else 'not_LA'
        mapped_predictions.append(prediction_str)
    
    return mapped_predictions

def paginate_csv(csv_path, page, page_size):
    df = pd.read_csv(csv_path)
    # Calculate start and end indices of the desired slice
    start = (page - 1) * page_size
    end = page * page_size
    # Slice the DataFrame
    paginated_df = df.iloc[start:end]
    # Convert to list of lists, include headers if it's the first page
    if start == 0:
        data = [df.columns.tolist()] + paginated_df.values.tolist()
    else:
        data = paginated_df.values.tolist()
    return data

def eval(url,project):
    if os.path.exists(f"{project}_eval.csv"):
        print(f"{project}_eval.csv found")
    else:
        print(f"{project}_eval.csv not found")
        if os.path.exists(f"{project}_data.csv"):
           print(f"{project}_data.csv found")
        else:
            repodownloader.extract_java_methods_from_project(url,project)
        file = f"{project}_data.csv"
        print(f"{file} found")
        model_path = f"model/buildship_final_model"
        if os.path.exists(model_path):
            print("Model found")
        else:
            train_combined_model.train("buildship")
            print("Model trained")
        df = pd.read_csv(file)
        df = df.drop(columns=['comments'])
        df['index']=df.index
        copy_df = df.copy()
        df = df.drop(columns=['path'])
        print("df.copied")
        df["is"] = 0
        df["set"] = 0
        df["get"] = 0
        df["DOS"] = 0
        predictor = ktrain.load_predictor(model_path)
        pred = predictor.predict(df, return_proba=True)
        pred = apply_threshold(pred)
        print(pred[0])
        labels = ['is', 'set', 'get', 'DOS']
        mapped_predictions = map_predictions(pred, labels)
        copy_df['LABEL'] = mapped_predictions
        pred = predictor.predict(df)
        copy_df['PREDICTION'] = pred
        copy_df.drop(columns=['index'],inplace=True)
        copy_df.to_csv(f"{project}_eval.csv",index=False)
    return f"{project}_eval.csv"

def csv_to_list(csv_path, page, page_size):
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    max_page = math.ceil(total_rows / page_size)
    start = (page - 1) * page_size
    end = page * page_size
    paginated_df = df.iloc[start:end]
    # Include headers only on the first page or if it's a single page
    # if start == 0 or max_page == 1:
    data = [df.columns.tolist()] + paginated_df.values.tolist()
    # else:
    #     data = paginated_df.values.tolist()
    return data, max_page

@app.route('/')
def home():
    return render_template('index_LA.html')

@app.route('/submit', methods=['POST','GET'])
def submit():
    p=None
    if request.method == 'POST':
        github_url = request.form['githubUrl']
        project_name = request.form['projectName']
        # Assume this function is implemented to download and return the path to the CSV
        print(f"project_name: {project_name} and github_url: {github_url}")
        csv_path = eval(github_url, project_name)
        session['csv_path'] = csv_path
        return redirect(url_for('submit', csv_path=csv_path))
    else:
        page = request.args.get('page', 1, type=int)
        page_size = 10  # Or another suitable size
        csv_path = session.get('csv_path', '')
        if not csv_path:
            # Handle error or redirect
            return "CSV path not provided", 400
        # Add logic to calculate the total number of rows in your CSV
        csv_content, max_page = csv_to_list(csv_path, page, page_size)
    return render_template('display_csv.html', csv_content=csv_content, current_page=page,max_page=max_page)

if __name__ == '__main__':
    app.run(debug=True,port=8080)
