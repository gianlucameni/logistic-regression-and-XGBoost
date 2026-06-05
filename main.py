from flask import Flask, jsonify, request
from it.valtellina.analyzer.analyzer import Analyzer
from it.valtellina.graphs.graphs import Graphs
from it.valtellina.splitter_train_test.splitter_train_test import SplitterTrainTest
from it.valtellina.one_hot_encoder.one_hot_encoder import OHEncoder
from it.valtellina.logistic_regression.logistic_regression import LogRegression
from it.valtellina.xgboost.xgboost_model import XGBoostModel

app = Flask(__name__)

@app.route('/')
def intro():
    return 'Welcome to Mushroom Classification Analyzer!'

    #stampare in home le route

@app.route('/api/missing-values')
def missing_values():
    msr = Analyzer() # creo oggetto che contiene il df mushrooms
    return jsonify({"missing values:": msr.missing_values()})

@app.route('/api/cleaning-data')
def cleaning_data():
    msr = Analyzer()
    msr.impute_categorical_unknown() # fix missing values
    return jsonify({
        "missing values": msr.missing_values(),
        "duplicates": msr.remove_duplicates()
    })




if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
