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

### Avvio in locale

Per visualizzare l'API in locale è necessario recarsi al seguente indirizzo:

   ```
   http://127.0.0.1:5000/
   ```
## Avvio con Docker

### Prerequisiti

Assicurati di avere installato Docker. Puoi scaricare l'applicazione [qui](https://www.docker.com/products/docker-desktop/).

Verifica l’installazione con:

```bash
docker --version
```

### Utilizzo di Docker

1. Crea una build da linea di comando:
   ```bash
   docker build -t mushroom-analyzer .
   ```
   
2. Controlla le informazioni del container creato con il comando:
   ```bash
   docker ps
   ```

3. Avvia l'applicazione da linea di comando, aggiungendo in fondo l'id del container:
   ```bash
   docker run -d --name mushroom-analyzer -p 5000:5000 
   ```

4. Accedi all'applicazione tramite il tuo browser all'indirizzo:
   ```
   http://127.0.0.1:5000/
   ```
