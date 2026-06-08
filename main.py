from flask import Flask, jsonify, request, send_file
from it.valtellina.analyzer.analyzer import Analyzer
from it.valtellina.graphs.graphs import Graphs
from it.valtellina.splitter_train_test.splitter_train_test import SplitterTrainTest
from it.valtellina.one_hot_encoder.one_hot_encoder import OHEncoder
from it.valtellina.logistic_regression.logistic_regression import LogRegression
from it.valtellina.xgboost.xgboost_model import XGBoostModel

app = Flask(__name__)

@app.route('/')
def intro():
    return """
    <h1>Mushroom Classification Analyzer API</h1>

    <h2>Available Endpoints</h2>

    <ul>
        <li><b>GET</b> /api/missing-values</li>
        <li><b>GET</b> /api/cleaning-data</li>
        <li><b>GET</b> /api/distribution-target</li>
        <li><b>GET</b> /api/correlation-target</li>
        <li><b>GET</b> /api/distribution-feature/&lt;feature&gt;</li>
        <li><b>GET</b> /api/logistic-regression</li>
        <li><b>GET</b> /api/XGBoost</li>
    </ul>

    <h2>Examples</h2>

    <ul>
        <li>/api/distribution-feature/odor</li>
        <li>/api/distribution-feature/cap-shape</li>
        <li>/api/distribution-feature/habitat</li>
    </ul>
    """



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

@app.route('/api/distribution-target')
def distribution_target():
    msr = Analyzer()
    msr.impute_categorical_unknown()
    msr.drop_columns(["veil-type"])  # valore costante

    # Trasformiamo il target in binario 0-1. edible 0, poisonous 1
    msr.prepare_target("poisonous")

    grf = Graphs(msr.df)
    img = grf.plot_target("poisonous") # --> target bilanciato

    return f"""<img src="data:image/png;base64,{img}" />"""

@app.route('/api/correlation-target')
def correlation_target():
    msr = Analyzer()
    msr.impute_categorical_unknown()
    msr.drop_columns(["veil-type"])  # valore costante

    # Trasformiamo il target in binario 0-1. edible 0, poisonous 1
    msr.prepare_target("poisonous")

    splitter = SplitterTrainTest(msr.df, "poisonous")
    X_train, X_test, y_train, y_test = splitter.split()

    # encoding
    ohe = OHEncoder(X_train, X_test)
    X_train_encoded, X_test_encoded = ohe.encode()

    # correlazione con il target
    corr = Graphs(X_train_encoded)
    img = corr.correlation_target(y_train)

    return f"""<img src="data:image/png;base64,{img}" />"""


@app.route('/api/distribution-feature/<feature>', methods=['GET', 'POST'])
def distribution_feature(feature):
    msr = Analyzer()
    msr.impute_categorical_unknown()

    #data = request.json
    #feature = data.get("feature")  # --> feature : name

    if feature not in msr.df.columns.tolist():
        return jsonify({"message": "feature non presente"})

    #msr.drop_columns(["veil-type"])  # valore costante

    # Trasformiamo il target in binario 0-1. edible 0, poisonous 1
    msr.prepare_target("poisonous")

    grf = Graphs(msr.df)
    img = grf.plot_categorical(feature, "poisonous")

    #return f"""<img src="data:image/png;base64,{img}" />"""
    return send_file(img, mimetype='image/png')


@app.route('/api/logistic-regression')
def logistic_regression():
    msr = Analyzer()
    msr.impute_categorical_unknown()
    msr.drop_columns(["veil-type"])  # valore costante

    msr.prepare_target("poisonous")

    # splitting train e test set
    splitter = SplitterTrainTest(msr.df, "poisonous")
    X_train, X_test, y_train, y_test = splitter.split()

    # encoding
    ohe = OHEncoder(X_train, X_test)
    X_train_encoded, X_test_encoded = ohe.encode()

    # applichiamo la regressione logistica
    log_reg_model = LogRegression()

    log_reg_model.fit(X_train_encoded, y_train)

    y_pred = log_reg_model.predict(X_test_encoded)

    metrics = log_reg_model.metrics(X_test_encoded, y_test)

    return jsonify({"metriche": metrics})


@app.route('/api/XGBoost')
def xgboost():
    msr = Analyzer()
    msr.impute_categorical_unknown()
    msr.drop_columns(["veil-type"])  # valore costante

    msr.prepare_target("poisonous")

    # splitting train e test set
    splitter = SplitterTrainTest(msr.df, "poisonous")
    X_train, X_test, y_train, y_test = splitter.split()

    # encoding
    ohe = OHEncoder(X_train, X_test)
    X_train_encoded, X_test_encoded = ohe.encode()

    # applichiamo XGBoost con OHE
    xgb_model = XGBoostModel()

    xgb_model.fit(X_train_encoded, y_train)

    y_pred_xgb = xgb_model.predict(X_test_encoded)

    metrics = xgb_model.metrics(X_test_encoded, y_test)

    return jsonify({"metriche": metrics})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
