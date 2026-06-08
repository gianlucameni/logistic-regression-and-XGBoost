import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import base64
import io

class Graphs:
    def __init__(self, df):
        self.df = df

    def plot_categorical(self, col, target='class'):
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        # 1. COUNT PLOT
        sns.countplot(x=col, data=self.df, ax=ax[0])
        ax[0].set_title(f"Count plot - {col}")

        # 2. PERCENTUALI
        tab = pd.crosstab(self.df[col], self.df[target], normalize='index')
        tab.plot(kind='bar', stacked=True, ax=ax[1])
        ax[1].set_title(f"Percentuali - {col}")
        ax[1].legend(title=target)

        plt.tight_layout()
        #plt.show()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        #img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        return buf

    def plot_target(self, col):
        fig, ax = plt.subplots(1, 1, figsize=(14, 5))

        # COUNT PLOT
        sns.countplot(x=col, data=self.df, ax=ax)
        ax.set_title(f"Count plot - {col}")

        #plt.show()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        return img_base64

    def correlation_target(self, y_train):
        # copia dati + target
        df_corr = self.df # X_train
        df_corr["target"] = y_train

        # correlazione con target
        corr_target = df_corr.corr()["target"].drop("target")

        # prendo le TOP 20 (in valore assoluto)
        top20 = corr_target.abs().sort_values(ascending=False).head(20)

        # ordino per visualizzazione
        top20 = corr_target.loc[top20.index].sort_values()

        # plot
        plt.figure(figsize=(10, 6))
        top20.plot(kind='barh')

        plt.title("Top 20 correlazioni con il target")
        plt.xlabel("Correlazione")
        #plt.show()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        return img_base64