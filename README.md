# Relazione Tecnica: Classificazione di Frodi Bancarie con Quantum SVM
### Pipeline QSVM sul Dataset BAF NeurIPS 2022 — Analisi completa del codice e dei risultati

---

## 0. Configurazione dell'Ambiente (WSL e Python)

Per eseguire correttamente la pipeline, specialmente per sfruttare l'accelerazione GPU con Qiskit, è fortemente raccomandato l'uso di **WSL (Windows Subsystem for Linux)** o un sistema operativo basato su Linux. Di seguito i passaggi per configurare l'ambiente:

### 1. Accesso a WSL e navigazione
Assicurati di aver installato WSL sul tuo sistema Windows. Apri il terminale WSL (es. Ubuntu) e spostati nella directory del progetto (se per esempio si trova sul disco D):
```bash
cd /mnt/d/MyProjects/quantum_computing_project
```

### 2. Installazione di Python e Virtual Environment
Assicurati di avere una versione di Python compatibile (es. Python 3.12). Aggiorna i pacchetti e installa i moduli necessari:
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev -y
```

### 3. Creazione e Attivazione dell'Ambiente Virtuale
Per prevenire problemi di permessi incrociati con il file system NTFS di Windows, crea l'ambiente virtuale all'interno del file system nativo di Linux (es. nella tua cartella home `~/.venvs`):
```bash
# Crea la cartella per gli ambienti virtuali (se non esiste)
mkdir -p ~/.venvs

# Crea l'ambiente virtuale specifico per il progetto
python3.12 -m venv ~/.venvs/quantum_env

# Attiva l'ambiente virtuale
source ~/.venvs/quantum_env/bin/activate
```
*(Nota: dovrai eseguire il comando di attivazione `source` ogni volta che apri un nuovo terminale).*

### 4. Installazione delle Dipendenze
Con l'ambiente attivato, installa le librerie necessarie. Per utilizzare l'accelerazione GPU NVIDIA, installa la versione corretta di Qiskit e il pacchetto `qiskit-aer-gpu`:
```bash
pip install "qiskit==1.1.1"
pip install qiskit-aer-gpu
pip install scikit-learn pandas numpy matplotlib seaborn imbalanced-learn
```

Una volta completata la configurazione, puoi eseguire comodamente i programmi Python all'interno di WSL (o usare gli script `run.sh` / `run.bat` forniti).

---

## 1. Introduzione e contesto

Il dataset **Bank Account Fraud (BAF)**, rilasciato al NeurIPS 2022, rappresenta un benchmark realistico per la rilevazione di frodi nell'apertura di conti bancari. Contiene circa 1 milione di richieste di conto corrente, di cui meno del 2% classificate come fraudolente, ed è caratterizzato da un forte sbilanciamento tra le classi e dalla presenza di valori mancanti codificati come `-1`.

L'obiettivo di questo lavoro è costruire una pipeline completa che:
1. Preprocessi il dataset in modo corretto e rigoroso
2. Riduca le feature da 28 a esattamente **4 dimensioni**, compatibili con una ZZFeatureMap a 4 qubit
3. Addestri un classificatore QSVM basato su kernel quantistico calcolato con il formalismo compute-uncompute
4. Confronti le prestazioni con baseline classiche (SVM Lineare e SVM RBF)

---

## 2. Architettura della pipeline

La pipeline è strutturata in 12 blocchi sequenziali. Lo schema logico è il seguente:

```
Base.csv
   │
   ▼
[Blocco 1] Caricamento e ispezione
   │
   ▼
[Blocco 2] Preprocessing
   ├── Drop: device_fraud_count, month
   ├── LabelEncoding colonne categoriche
   ├── Imputazione -1 con mediana
   └── StandardScaler (µ=0, σ=1)
   │
   ▼
[Blocco 3] Mutual Information (diagnostico)
   │
   ▼
[Blocco 4] PCA → 4 componenti (= 4 qubit)
   │
   ▼
[Blocco 5] Bilanciamento classi (150 per classe → 300 campioni)
   │
   ▼
[Blocco 6] Scaling angolare in [-π, +π]
   │
   ▼
[Blocco 7] Train/Test split 75/25 stratificato
   │
   ├──────────────────────────────────┐
   ▼                                  ▼
[Blocco 8] ZZFeatureMap            [Blocco 10] SVM Lineare + SVM RBF
Kernel quantistico compute-uncompute   (baseline classica)
   │
   ▼
[Blocco 9] SVC (kernel='precomputed')
   │
   ▼
[Blocco 11] Visualizzazioni e confronto
   │
   ▼
[Blocco 12] Salvataggio risultati
```

---

## 3. Dettaglio del preprocessing (Blocchi 1–2)

### 3.1 Ispezione del dataset

Il dataset originale presenta 30 colonne, di cui 28 feature, la variabile target `fraud_bool` e due colonne eliminate: `device_fraud_count` (contiene informazioni sulle frodi future, fonte di data leakage) e `month` (variabile temporale non predittiva nella forma in cui è codificata).

### 3.2 Encoding e imputazione

Le colonne categoriche (`payment_type`, `source`, `device_os`, `employment_status`) vengono trasformate tramite `LabelEncoder`. I valori `-1`, usati nel dataset originale per indicare valori mancanti nelle colonne `prev_address_months_count`, `current_address_months_count`, `intended_balcon_amount` e `bank_months_count`, vengono sostituiti con la **mediana della colonna calcolata escludendo i -1 stessi**. Questa scelta è corretta metodologicamente: usare i -1 nel calcolo della mediana sposterebbe verso il basso la stima del valore tipico.

### 3.3 StandardScaler

Tutte le feature numeriche vengono standardizzate con `StandardScaler` prima del calcolo della Mutual Information e della PCA. Senza questo passaggio, feature con range molto ampi (es. `zip_count_4w` fino a 6.700) avrebbero dominato la decomposizione PCA, alterando i risultati indipendentemente dalla loro reale informatività.

---

## 4. Feature selection: Mutual Information (Blocco 3)

La Mutual Information (MI) misura la dipendenza statistica tra ciascuna feature e il target `fraud_bool`, catturando relazioni sia lineari sia non lineari. Il calcolo è effettuato sulle feature standardizzate con `mutual_info_classif` di scikit-learn.

![Grafico MI Scores](mi_scores.jpg)

I punteggi mostrano una chiara gerarchia:

| Feature | MI Score | Interpretazione |
|---|---|---|
| `email_is_free` | ~0.119 | Forte predittore: le frodi usano più spesso email gratuite |
| `has_other_cards` | ~0.114 | Indicatore di profilo finanziario |
| `keep_alive_session` | ~0.113 | Comportamento anomalo di sessione |
| `phone_home_valid` | ~0.112 | Validità del numero di telefono fisso |
| `proposed_credit_limit` | ~0.080 | Limite di credito richiesto |
| `phone_mobile_valid` | ~0.072 | Validità numero mobile |
| `device_os` | ~0.070 | Sistema operativo del dispositivo |

Le prime 4 feature hanno punteggi molto simili tra loro (0.11–0.12), suggerendo che il segnale predittivo è distribuito su più variabili e non concentrato su una sola. Le feature in fondo alla lista (es. `device_distinct_emails_8w`, `prev_address_months_count`) contribuiscono in modo trascurabile.

> **Ruolo della MI nella pipeline finale**: in questa versione del codice la MI è usata in modo puramente diagnostico, per comprendere quali feature portano informazione sul target. La proiezione finale verso i 4 qubit viene effettuata interamente tramite PCA, che è una trasformazione lineare non supervisionata. Una versione migliorata potrebbe integrare la feature con MI massima (es. `email_is_free`) come quarta dimensione in luogo della 4ª componente principale.

---

## 5. Feature projection: PCA a 4 componenti (Blocco 4)

La PCA proietta le 28 feature standardizzate su 4 assi ortogonali che massimizzano la varianza spiegata, riducendo lo spazio di input alle esatte 4 dimensioni richieste dalla `ZZFeatureMap`.

![Grafico Scree Plot PCA](pca_scree-2.jpg)

### 5.1 Varianza spiegata

| Componente | Varianza individuale | Varianza cumulativa |
|---|---|---|
| PC1 | 8.9% | 8.9% |
| PC2 | 6.8% | 15.7% |
| PC3 | 5.7% | 21.4% |
| PC4 | 5.6% | 27.0% |

La varianza cumulativa al 27.0% è bassa ma attesa: il dataset BAF è ad alta dimensionalità e le feature sono debolmente correlate tra loro, per cui nessuna componente cattura una porzione ampia di varianza. Questo implica che la compressione sacrifica circa il 73% dell'informazione originale, un costo inevitabile nel vincolo dei 4 qubit.

### 5.2 Distribuzione dei campioni nello spazio PCA

![Scatter PC1 vs PC2](scatter_pc1_pc2-1.jpg)

Lo scatter plot mostra i 300 campioni bilanciati (150 legittimi, 150 frodi) nello spazio PC1–PC2, già riscalati in radianti. Le due classi risultano **fortemente sovrapposte**: non esiste un confine lineare netto, e le frodi (in rosso) sono distribuite in modo irregolare su tutto il dominio. Questa osservazione è fondamentale per interpretare i risultati dei classificatori: il problema è genuinamente difficile in questo spazio proiettato.

---

## 6. Bilanciamento e scaling angolare (Blocchi 5–6)

### 6.1 Bilanciamento

Il dataset originale è fortemente sbilanciato (~2% di frodi). Per addestrare la QSVM in modo corretto e senza che il classificatore ignori la classe minoritaria, vengono selezionati 150 campioni per classe tramite undersampling (classe maggioritaria) e oversampling con rimpiazzo se necessario (classe minoritaria). Il campione totale è quindi 300 osservazioni bilanciate 50/50.

La dimensione di 300 campioni è imposta dal costo computazionale: calcolare la matrice kernel con il metodo compute-uncompute richiede \( \frac{N(N+1)}{2} \) esecuzioni di circuito per il solo training set. Con 225 campioni di training (75% di 300), ciò corrisponde a circa **25.425 circuiti quantistici**, ognuno simulato con 8.192 shots.

### 6.2 Scaling angolare

Le 4 componenti PCA vengono riscalate da `MinMaxScaler` nell'intervallo \( [-\pi, +\pi] \). Questo passaggio è **critico per la correttezza fisica del circuito**: la `ZZFeatureMap` usa i valori di input come angoli di rotazione nelle porte di fase \( P(\theta) \). Senza questo scaling, i valori PCA (dell'ordine di ±3 float) potrebbero coincidentalmente restare in un range accettabile, ma in generale si rischierebbe di avere angoli troppo grandi o troppo piccoli, compromettendo l'espressività del circuito.

---

## 7. Circuito quantistico: ZZFeatureMap (Blocco 8)

### 7.1 Struttura del circuito

La `ZZFeatureMap` con `feature_dimension=4`, `reps=1` ed `entanglement='full'` produce il circuito seguente (output Qiskit):

```
q_0: ┤ H ├┤ P(2x[0]) ├──■────────────────────────────────────■────■──...
q_1: ┤ H ├┤ P(2x[1]) ├┤ X ├┤ P(2(π-x[0])(π-x[1])) ├┤ X ├──┼────...
q_2: ┤ H ├┤ P(2x[2]) ├─────────────────────────────────┤ X ├┤ P(2(π-x[0])(π-x[2])) ├...
q_3: ┤ H ├┤ P(2x[3]) ├──────────────────────────────────────────────────...
```

Il circuito si compone di tre tipi di gate:

1. **Hadamard (H)**: porta ogni qubit in una sovrapposizione uniforme \( |+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \)
2. **Fasi singole \( P(2x_i) \)**: codificano il valore della feature \( x_i \) come rotazione di fase sul qubit \( i \)
3. **Fasi entangled \( P(2(\pi - x_i)(\pi - x_j)) \)** applicate tramite coppie CNOT: codificano le **interazioni quadratiche** tra feature diverse, creando correlazioni quantistiche tra i qubit

### 7.2 Mappa di feature indotta

La funzione di encoding produce lo stato:
\[ |\Phi(x)\rangle = U_{\Phi}(x)|0\rangle^{\otimes 4} \]

dove \( U_{\Phi}(x) \) è il circuito descritto. Con `entanglement='full'` vengono create tutte le \( \binom{4}{2} = 6 \) interazioni a coppie, rendendo il circuito più espressivo rispetto all'entanglement lineare (solo 3 coppie).

### 7.3 Calcolo del kernel: metodo compute-uncompute

Il kernel quantistico è calcolato manualmente con il paradigma compute-uncompute:

\[ K(x_i, x_j) = |\langle 0 | U_{\Phi}^\dagger(x_j) U_{\Phi}(x_i) | 0 \rangle|^2 \]

Il circuito eseguito per ogni coppia è:

```
|0...0⟩ → U_Φ(xi) → U_Φ†(xj) → misura P(|0000⟩)
```

La probabilità misurata di osservare lo stato \( |0000\rangle \) dopo questo circuito è la **fidelity** tra i due stati quantistici \( |\Phi(x_i)\rangle \) e \( |\Phi(x_j)\rangle \), che funge da misura di similarità (valore del kernel). Questo kernel è per costruzione **simmetrico** e **definito positivo**, caratteristiche necessarie per l'utilizzo con SVM.

### 7.4 Matrice kernel quantistica

![Matrice Kernel Quantistica](kernel_matrix-1.jpg)

La heatmap della matrice kernel \( K_{train} \) (225×225 campioni) mostra:

- **Diagonale a 1.0** (scuro): ogni campione è identico a sé stesso, \( K(x_i, x_i) = 1 \) per definizione
- **Valori fuori diagonale molto bassi** (~0.0–0.05, colore giallo chiaro): la maggior parte delle coppie produce una fidelity quasi nulla
- **Poche strutture locali**: sporadici punti più chiari distribuiti casualmente, senza blocchi di similarità strutturati

Questo pattern è interpretabile come una **matrice quasi-diagonale**, in cui il kernel distingue formalmente ogni campione da tutti gli altri, ma non riesce a costruire regioni di similarità coerenti e discriminative. In pratica, l'embedding quantistico sta proiettando i dati in uno spazio molto ad alta dimensione (lo spazio di Hilbert a 4 qubit, di dimensione \( 2^4 = 16 \)) in modo eccessivamente dispersivo, riducendo la capacità discriminativa del classificatore.

---

## 8. Addestramento e risultati (Blocchi 9–10)

### 8.1 SVM con kernel precomputed

La matrice \( K_{train} \) viene passata direttamente come matrice di Gram a `sklearn.svm.SVC(kernel='precomputed', C=1.0)`. Il classificatore opera esclusivamente sulle similarità quantistiche, senza mai vedere le feature originali né le componenti PCA.

### 8.2 Confronto delle metriche

![Confronto metriche benchmark](benchmark_metrics-1.jpg)

| Modello | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **QSVM (ZZFeatureMap)** | 0.520 | 0.513 | 0.541 | 0.526 | 0.585 |
| **SVM Lineare** | 0.613 | 0.611 | 0.595 | 0.603 | 0.709 |
| **SVM RBF** | **0.693** | **0.684** | **0.703** | **0.693** | **0.712** |

La SVM RBF domina su tutte le metriche. La SVM lineare si colloca a metà. La QSVM è il modello con prestazioni più basse, con un'Accuracy di 0.52 che è solo marginalmente superiore alla classificazione casuale (0.50 su dataset bilanciato), e un ROC-AUC di 0.585 che indica una capacità discriminativa molto debole.

### 8.3 Confusion matrices

![Confusion Matrices](confusion_matrices.jpg)

Le confusion matrix sul test set (50 campioni, 25 per classe) evidenziano il comportamento di ciascun modello:

| Modello | Veri Positivi (Frodi) | Veri Negativi (Legit.) | Falsi Positivi | Falsi Negativi |
|---|---|---|---|---|
| QSVM | 20 | 19 | 19 | 17 |
| SVM Lineare | 22 | 24 | 14 | 15 |
| SVM RBF | 26 | 26 | 12 | 11 |

- La **QSVM** classifica 39 campioni su 50 correttamente, ma con un pattern quasi casuale: sbaglia quasi la metà dei casi per entrambe le classi
- La **SVM Lineare** migliora sensibilmente, soprattutto nella classe legittima, riducendo i falsi positivi da 19 a 14
- La **SVM RBF** è il modello più bilanciato: raggiunge 26/25 su entrambe le classi, con il minor numero di errori sia per i falsi positivi sia per i falsi negativi

### 8.4 Decision boundaries

![Decision Boundaries](decision_boundaries-1.jpg)

Il grafico delle decision boundaries (calcolate sulle prime 2 componenti PCA) mostra:

- **SVM Lineare**: frontiera retta, separa il piano in due semipiani con una diagonale che tende a privilegiare la classe frode in basso a destra
- **SVM RBF**: frontiera curva e più adattiva, che si piega attorno a cluster locali; la maggiore flessibilità spiega il miglioramento delle prestazioni
- **QSVM (approx 2D)**: frontiera non lineare con struttura irregolare; la visualizzazione è però un'**approssimazione** calcolata con un kernel RBF classico sulla griglia 2D, non con il vero kernel quantistico 4D, quindi deve essere interpretata solo qualitativamente

Il grafico conferma visivamente la forte sovrapposizione delle classi: nessun classificatore riesce a delimitare regioni pulite di frodi e legittimi, il che è consistente con le metriche numeriche.

---

## 9. Analisi critica dei risultati

### 9.1 Punti di forza

**Correttezza del workflow**: la pipeline è implementata in modo metodologicamente rigoroso. Il preprocessing elimina correttamente le fonti di leakage, l'imputazione è robusta, la standardizzazione precede la MI e la PCA, e lo scaling angolare è calibrato per la fisica della feature map.

**Scenario difficile e realistico**: le due classi si sovrappongono nello spazio PCA, rendendo questo un problema genuinamente difficile. Questo è preferibile a configurazioni artificialmente separabili, perché permette una valutazione più onesta della generalizzazione.

**Confronto con baseline**: la presenza di SVM Lineare e RBF come riferimento permette di contestualizzare le prestazioni della QSVM, evitando di presentare risultati assoluti privi di significato comparativo.

**Kernel quantistico manuale**: l'implementazione compute-uncompute è corretta e trasparente, con verifica della simmetria e della diagonale unitaria.

### 9.2 Limiti e problemi identificati

**Perdita di informazione troppo alta (27% di varianza spiegata)**: la compressione da 28 a 4 dimensioni tramite PCA è estrema. La maggior parte del segnale discriminativo viene sacrificata prima di entrare nel circuito quantistico. Questo è probabilmente il limite principale: la `ZZFeatureMap` lavora su input già molto degradati, il che rende difficile anche per un kernel non lineare ricavare strutture utili.

**Mutual Information non integrata nella proiezione finale**: il grafico MI mostra chiaramente che `email_is_free` è la feature singola più informativa, con un punteggio di ~0.119. Tuttavia questa feature non viene preservata esplicitamente nella proiezione finale: entra nella PCA insieme a tutte le altre 27 feature, e il suo contributo viene diluito nelle componenti principali, che massimizzano varianza e non correlazione con il target.

**Matrice kernel quasi-diagonale**: la heatmap mostra che il kernel quantistico produce similarità quasi nulle tra campioni diversi. Questo fenomeno, noto come **concentration of measure** o **kernel concentration**, si verifica quando la feature map è troppo espressiva e proietta punti vicini in stati quantistici quasi ortogonali, rendendo il kernel poco informativo. Con `reps=1` e `entanglement='full'` si è cercato di mitigare questo problema rispetto a `reps=2`, ma il risultato suggerisce che l'espressività rimane eccessiva rispetto alla struttura dei dati.

**Dimensione del campione limitata**: 225 campioni di training (150 train + 75 test no, 225 train + 75 test) è un numero piccolo per un dataset di 1 milione di righe. Le stime delle metriche hanno un'elevata varianza statistica, e piccole variazioni nel campionamento possono cambiare significativamente i risultati. Le conclusioni sono quindi indicative e non definitive.

**Decision boundary quantistica approssimata**: la visualizzazione della QSVM usa un kernel RBF classico per costruire la frontiera sulla griglia 2D. Questo introduce un'inconsistenza: ciò che viene visualizzato non è la vera superficie decisionale del modello quantistico.

**Costo computazionale sproporzionato**: il calcolo manuale di ~25.000 coppie con 8.192 shots ciascuna richiede tempi dell'ordine di minuti–ore su CPU. Per le stesse 225 osservazioni, SVM RBF è addestrata in millisecondi con risultati superiori.

---

## 10. Considerazioni sull'overfitting

Non emergono segnali forti di overfitting: le metriche sono basse sia per la QSVM sia, in misura minore, per le SVM classiche, il che indica che i modelli stanno apprendendo pattern generalmente poco strutturati e non memorizzando i dati di training. La SVM RBF ottiene un F1 di 0.693, che è consistente con un modello che generalizza parzialmente ma non perfettamente.

Il rischio più concreto non è l'overfitting classico ma il **sottoadattamento** (underfitting) della QSVM, che non riesce a costruire un margine utile a causa della matrice di Gram quasi-diagonale. In pratica, la QSVM sta facendo scelte di classificazione quasi casuali, non perché abbia memorizzato il training set, ma perché il kernel non le fornisce informazione strutturata sufficiente.

---

## 11. Suggerimenti per migliorare i risultati

1. **Ibrido PCA + MI**: usare 3 componenti PCA più la feature con MI massima (`email_is_free` o `has_other_cards`) come 4° input. Questo preserva una variabile altamente discriminativa in forma pura.

2. **Kernel concentration**: ridurre l'entanglement da `full` a `linear` può attenuare la dispersività del kernel. Alternativamente, ridurre la profondità del circuito non aiuta con `reps=1` ma si potrebbe esplorare una parametrizzazione manuale con angoli di fase ridotti.

3. **Ottimizzare C**: il valore `C=1.0` non è ottimizzato. Una grid search sul parametro di regolarizzazione potrebbe migliorare il margine di classificazione.

4. **Aumentare il campione**: anche passare da 150 a 300 campioni per classe (600 totali) aumenta il costo computazionale di circa 4×, ma potrebbe migliorare sensibilmente la stima del kernel e quindi le prestazioni.

5. **Feature engineering**: costruire nuove feature composte (es. rapporti tra variabili, interazioni tra le top-MI feature) prima della PCA potrebbe aumentare la varianza spiegata e la qualità delle componenti.

---

## 12. Conclusione

La pipeline implementata è tecnoicamente corretta e funziona end-to-end dalla preparazione del dato fino alla classificazione e alla generazione del report. I risultati mostrano però un chiaro divario tra la QSVM (Accuracy 0.52, ROC-AUC 0.585) e i modelli classici, specialmente la SVM RBF (Accuracy 0.693, ROC-AUC 0.712).

Questo risultato non invalida il lavoro: dimostra invece che il vantaggio quantistico non emerge automaticamente dall'uso di una feature map quantistica, specialmente quando il preprocessing comprime fortemente l'informazione (27% di varianza spiegata) e il kernel risultante è scarsamente strutturato. Ai fini della tesi, questo risultato è scientificamente onesto e correttamente motivato: mostra la pipeline funzionante, identifica i colli di bottiglia e indica percorsi concreti di miglioramento.

