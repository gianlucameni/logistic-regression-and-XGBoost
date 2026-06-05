import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.stats import jarque_bera
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
import base64
import io

from ucimlrepo import fetch_ucirepo

class Analyzer:

    def __init__(self, dataset_id=73):

        # download dataset
        dataset = fetch_ucirepo(id=dataset_id)

        # creo dataframe completo
        self.df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

    def show_head(self):
        print(self.df.head())


    # ----------------------------
    # INFO BASE
    # ----------------------------
    def overview(self):
        print("Shape:", self.df.shape)
        print("\nInfo:")
        print(self.df.info())

    # Missing values
    def missing_values(self):
        print(self.df.isnull().sum())

    # Imputo Unknown
    def impute_categorical_unknown(self):
        categorical_missing_cols = ["stalk-root"]
        for col in categorical_missing_cols:
            self.df[col] = self.df[col].fillna("Unknown")

    # ----------------------------
    # GESTIONE DUPLICATI
    # ----------------------------

    def remove_duplicates(self):
        before = self.df.shape[0]
        self.df = self.df.drop_duplicates()
        after = self.df.shape[0]
        print(f"Duplicati trovati e rimossi: {before - after}")

    # ----------------------------
    # OUTLIERS (IQR METHOD)
    # ----------------------------

    def prepare_target(self, target):
        self.df[target] = self.df[target].map({'e': 0, 'p': 1})

    def count_values(self):
        for col in self.df.columns:
            print(f"\n--- {col} ---")
            counts = self.df[col].value_counts(dropna=False)
            print(counts.sort_values(ascending=False))

    # describe
    def info_describe(self,column):
        print(self.df[column].describe())

    # ----------------------------
    # PLOTS (per colonna)
    # ----------------------------


    def drop_columns(self,columns):
        for col in columns:
            self.df = self.df.drop(col, axis=1)

    def correlation_matrix(self):
        # selezione colonne numeriche
        num_df = self.df.select_dtypes(include=['number'])
        # matrice di correlazione
        corr = num_df.corr()
        # maschera triangolo superiore
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5
        )

        plt.title("Correlation Heatmap (Lower Triangle)")
        #plt.show()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)

        img_base64 = base64.b64encode(buf.read()).decode("utf-8")

        plt.close()

        return img_base64

