from it.valtellina.analyzer.analyzer import Analyzer
from it.valtellina.graphs.graphs import Graphs
from it.valtellina.splitter_train_test.splitter_train_test import SplitterTrainTest
from it.valtellina.one_hot_encoder.one_hot_encoder import OHEncoder

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

ohe.stampa(X_train_encoded)
ohe.stampa(X_test_encoded)
# gestire le features categoriche con one-hot encoding

#correlazione con il target
corr = Graphs(X_train_encoded)
corr.correlation_target(y_train)
# odor impatta di piu, sia positivamente(f, verso 1) sia negativamente(n, verso 0)