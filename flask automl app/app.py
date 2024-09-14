from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def build_model(df, split_size, seed_number):
    # Data split
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100, random_state=seed_number)

    reg = LazyRegressor(verbose=0, ignore_warnings=False)
    models_train, predictions_train = reg.fit(X_train, X_train, Y_train, Y_train)
    models_test, predictions_test = reg.fit(X_train, X_test, Y_train, Y_test)

    return predictions_train, predictions_test

def generate_plot(data, metric, orientation='horizontal'):
    plt.figure(figsize=(10, 6) if orientation == 'horizontal' else (6, 10))
    sns.set_theme(style="whitegrid")
    
    if orientation == 'horizontal':
        sns.barplot(x=metric, y=data.index, data=data)
    else:
        sns.barplot(y=metric, x=data.index, data=data)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    split_size = int(request.form['split_size'])
    seed_number = int(request.form['seed_number'])
    
    if 'file' in request.files:
        file = request.files['file']
        df = pd.read_csv(file)
    else:
        df = pd.read_csv("boston_house_dataset.csv")
    
    predictions_train, predictions_test = build_model(df, split_size, seed_number)

    r2_plot = generate_plot(predictions_test, 'R-Squared', 'horizontal')
    rmse_plot = generate_plot(predictions_test, 'RMSE', 'horizontal')
    time_plot = generate_plot(predictions_test, 'Time Taken', 'horizontal')

    return render_template('result.html', 
                           predictions_train=predictions_train, 
                           predictions_test=predictions_test,
                           r2_plot=r2_plot, 
                           rmse_plot=rmse_plot, 
                           time_plot=time_plot)

if __name__ == '__main__':
    app.run(debug=True)
