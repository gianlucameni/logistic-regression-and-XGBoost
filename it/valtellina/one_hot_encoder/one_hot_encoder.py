from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class OHEncoder():
    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test
    def encode(self):
        # Encoder
        encoder = OneHotEncoder(
            drop='first',  #elimina una dummy per ogni variabile categorica
                            # evitando multicollinearità nella regressione logistica.
            sparse_output=False,
            handle_unknown='ignore' # evita errori se nel test compare
                                    # una categoria non presente nel training set.
        )

        # Fit sul train, transform su train e test
        X_train_encoded = encoder.fit_transform(self.X_train)
        X_test_encoded = encoder.transform(self.X_test)

        # Conversione in DataFrame
        feature_names = encoder.get_feature_names_out()

        X_train_encoded = pd.DataFrame(
            X_train_encoded,
            columns=feature_names,
            index=self.X_train.index
        )

        X_test_encoded = pd.DataFrame(
            X_test_encoded,
            columns=feature_names,
            index=self.X_test.index
        )

        return X_train_encoded, X_test_encoded


    def stampa(self, df):
        print(df.head())