from it.valtellina.analyzer.analyzer import Analyzer
from it.valtellina.graphs.graphs import Graphs
from it.valtellina.splitter_train_test.splitter_train_test import SplitterTrainTest
from it.valtellina.one_hot_encoder.one_hot_encoder import OHEncoder
from it.valtellina.logistic_regression.logistic_regression import LogRegression
from it.valtellina.xgboost.xgboost_model import XGBoostModel

# creiamo oggetto che contiene il df
df = Analyzer()

#df.show_head()
#df.overview() # --> stalk-root ha 2.48k null
df.missing_values()
# riempiamo i missing con una nuova feature unknown
df.impute_categorical_unknown()
df.missing_values()

# andiamo a rimuovere eventuali duplicati
df.remove_duplicates() # 0 duplicati trovati e rimossi

# stampiamo grafici count plot
#grf = Graphs(df.df)
#grf.plot_categorical("cap-shape", "poisonous")
#grf.plot_target("poisonous") # --> target bilanciato



# Trasformiamo il target in binario 0-1. edible 0, poisonous 1
df.prepare_target("poisonous")

df.count_values()

df.drop_columns(["veil-type"]) # valore costante

# split -> encoding -> correlation
# splitting
# splitting train e test set
splitter = SplitterTrainTest(df.df, "poisonous")
X_train, X_test, y_train, y_test = splitter.split()

# encoding
ohe = OHEncoder(X_train, X_test)
X_train_encoded, X_test_encoded = ohe.encode()

#X_train_encoded.head()

#ohe.stampa(X_train_encoded)
#ohe.stampa(X_test_encoded)
# gestire le features categoriche con one-hot encoding

#correlazione con il target
#corr = Graphs(X_train_encoded)
#corr.correlation_target(y_train)
# odor impatta di piu, sia positivamente(f, verso 1) sia negativamente(n, verso 0)

# l'uso di drop = first nell'encoding permette di rimuovere la multicollinearità perfetta,
# successivamente essa viene gestita anche dalla regressione logistica
# che regola i pesi (di default usa penalty = L2, con C = 1)

#applichiamo la regressione logistica
log_reg_model = LogRegression()

log_reg_model.fit(X_train_encoded, y_train)

y_pred = log_reg_model.predict(X_test_encoded)

metrics = log_reg_model.metrics(X_test_encoded, y_test)

print("Logistic Regression")
print(metrics)

#applichiamo XGBoost con OHE
xgb_model = XGBoostModel()

xgb_model.fit(X_train_encoded, y_train)

y_pred_xgb = xgb_model.predict(X_test_encoded)

print("XGBoost Encoded")
xgb_model.metrics(X_test_encoded, y_test)

