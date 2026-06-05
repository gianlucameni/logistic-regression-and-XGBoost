"# logistic-regression-and-XGBoost" 

# Mushroom Classification Analyzer

Mushroom Classification Analyzer è un'API in Flask progettata per analizzare il dataset Mushroom e prevedere se un fungo è commestibile o velenoso utilizzando modelli di Machine Learning, in particolare Logistic Regression e XGBoost.

Il sistema include una pipeline completa di preprocessing dei dati categorici (One-Hot Encoding), analisi esplorativa del dataset e training di modelli supervisionati per la classificazione binaria.

## Funzionalità Principali

- **Analisi del dataset**: 
Esplorazione delle feature categoriche del dataset Mushroom, 
analisi delle distribuzioni e delle frequenze delle classi, 
studio delle relazioni tra variabili (EDA)
- **Processing dei dati**: 
Gestione dei valori mancanti (es. “Unknown”),
One-Hot Encoding delle variabili categoriche,
preparazione del dataset per modelli ML
- **Utilizzo di modelli di Machine Learning**:
Addestramento di modelli di Logistic Regression
Addestramento di modelli XGBoost
Valutazione delle performance tramite metriche standard (Accuracy, Precision, Recall, F1-score)

## Avvio di Mushroom Classificator Analyzer

### Requisiti

Prima di avviare Mushroom Classificator Analyzer, assicurati di avere installato tutte le dipendenze necessarie. Puoi trovarle nel file `requirements.txt`.
