# Mushroom Classification Analyzer

Mushroom Classification Analyzer è un'API in Flask 
progettata per analizzare il [dataset Mushroom](https://archive.ics.uci.edu/dataset/73/mushroom) e 
prevedere se un fungo è commestibile o velenoso utilizzando modelli di Machine Learning, 
in particolare Logistic Regression e XGBoost.

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

1. Costruisci l'immagine Docker da linea di comando:
   ```bash
   docker build -t mushroom-analyzer .
   ```
   
2. Si possono controllare le informazioni dell'immagine appena creata con il comando:
   ```bash
   docker image ls
   ```

3. Costruisci e avvia il container da linea di comando (copiando questo comando, verrà chiamato "mushroom-test"):
   ```bash
   docker run -d --name mushroom-test -p 5000:5000 mushroom-analyzer
   ```
   Se l'operazione è andata a buon fine è possibile vedere lo stavo attivo del container tramite il comando:
   ```bash
   docker ps
   ```

4. Accedi all'applicazione (nella sua route home) tramite il tuo browser all'indirizzo:
   ```
   http://127.0.0.1:5000/
   ```
## Utilizzo dell'API

L'API è consultabile direttamente da browser.

Se dovesse servire, è possibile installare dei plug-in, tra cui:

- Rest-Client (Chrome): [download](https://chromewebstore.google.com/detail/rest-client/oienkoejnhkbcibhdnpjoemdnmiokgah)
- Rested (Firefox): [download](https://addons.mozilla.org/en-US/firefox/addon/rested/)

L'utilizzo dei plug-in non permette però la restituzione delle immagini, facendo risultare "strana" la risposta di alcuni endpoint

## Funzioni di MCA

MCA è provvisto di diversi endpoint GET, consultabili nella route **home**.
```
http://127.0.0.1:5000/
```
Le funzionalità dell'API sono le seguenti:

### Missing Values

Endpoint che permette di visualizzare la quantità di missing values presente nel dataset
```
http://127.0.0.1:5000/api/missing-values
```

### Cleaning Data

Endpoint che permette di pulire il dataset riempiendo i valori nulli e droppando le colonne in eccesso per la creazione dei modelli
```
http://127.0.0.1:5000/api/cleaning-data
```

### Distribution Target

Endpoint che restituisce la distribuzione della variabile target "poisonous"
```
http://127.0.0.1:5000/api/distribution-target
```

### Correlation Target

Endpoint che restituisce la correlazione delle features (trattate con one-hot-encoded) con il target
```
http://127.0.0.1:5000/api/correlation-target
```

### Distribution Feature

Endpoint che restituisce la distribuzione della feature passata in input dopo l'ultimo " / "
```
http://127.0.0.1:5000/api/distribution-feature/<feature>
```

Un esempio corretto è il seguente:
```
http://127.0.0.1:5000/api/distribution-feature/odor
```

### Logistic Regression

Endpoint che stampa a video le metriche della logistic regression eseguita sul dataset ripulito
```
http://127.0.0.1:5000/api/logistic-regression
```

### XGBoost

Endpoint che stampa a video le metriche di XGBoost eseguito sul dataset ripulito
```
http://127.0.0.1:5000/api/XGBoost
```