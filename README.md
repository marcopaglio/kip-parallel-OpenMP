# kip-parallel-OpenMP
Parallelized version of Kernel Image Processing, i.e. kernel filtering applied to 2D images, with OpenMP.



## Complessità del problema

Siano $M$ e $N$ le dimensioni dell'immagine su cui applicare la convoluzione col kernel quadrato di dimensione $K$, la complessità del problema di Kernal Image Processing è $O(MNK^2)$.



## Processo di parallelizzazione

Parallelizzare un programma sequenziale non significa solo “aggiungere thread”, ma richiede un percorso metodico per evitare false parallelizzazioni, o addirittura rallentamenti, capire dove concentrare sforzi e analisi col fine di ottenere miglioramenti, e capire **quanto bene l’algoritmo** sia stato parallelizzato e quanto si possa ancora fare. A tal proposito, il workflow seguito in questo lavoro può essere così riassunto:

  1. **Profiling del codice sequenziale** (e.g. tramite VTune) per individuare dove intervenire:
       * Identificare gli *hotspot*, i.e. funzioni che consumano più tempo.
       * Capire se il collo di bottiglia è CPU, memoria, o I/O.
       
  2. **Inserire parallelismo incrementale**:
       * Riconoscere sezioni parallele
       * Privatizzare le variabili
       * Scegliere scheduling
       
  3. **Analisi della scalabilità teorica** fino a saturazione delle risorse:
       * **Legge di Amdahl**: valutare il limite teorico dato dalla frazione sequenziale, mediante [*strong scaling*](#strong-scaling).
       * **Legge di Gustafson**: valutare la bontà della parallelizzazione al crescere del problema, mediante [*weak scaling*](#weak-scaling).
       
  4. **Profiling del codice parallelo** mirato ai problemi che hanno generato comportamenti inattesi (teoria $\neq$ pratica) sugli scaling, col fine di:
       * **Quantificare l’efficacia della parallelizzazione**: quanto spingersi con il numero di core.
       * **Individuare colli di bottiglia e problematiche**: memoria, load balance, sincronizzazioni, comunicazioni, etc.
       * **Guidare lo sviluppo**: capire se conviene lavorare sugli overhead, oppure ridisegnare l’algoritmo.
       
       |    Analysis       |      Cosa cercare       |        Possibili cause       |     Azioni consigliate     |
       |:-----------------:|:-----------------------:|:----------------------------:|:--------------------------:|
       | **Hotspot**       | sezioni dominanti       | *codice sequenziale*         | parallelizzare             |
       |                   |                         |                              |  ridisegnarre algoritmo    |
       |                   |                         |                              |                            |
       | **Threading**     | thread in idle          | *Load imbalance*             |             scheduling     |
       |                   |                         | *Sync overhead*              | rimuovere/ridurre barriere |
       |                   |                         |                              |                            |
       | **Memory Access** | banda di memoria satura | *Bottleneck memoria*         | migliorare locality        |
       |                   |  poor core utilization  |                              | tiling/blocking            |
       |                   |                         |                              | migliorare uso cache       |
       |                   |                         |                              | ridurre traffico memoria   |
  
  5. **Ripetere** tuning > scaling > profiling finché non si raggiunge un compromesso accettabile.


### Strong Scaling

Lo **strong scaling** serve a capire se il programma può diventare più veloce.

Una volta fissata la dimensione del problema, all'aumentare del numero di core $p$ vengono calcolate le seguenti metriche:
  * **Speedup**: $S(p) = \frac{T(1)}{T(p)}$, dove $T(p)$ è il tempo misurato con $p$ core.
  * **Efficienza strong**: $E(p) = \frac{S(p)}{p}$.
  * **Serial fraction di Karp–Flatt**: $f(p) = \frac{1/S(p) - 1/p}{1 - 1/p}$ misura la porzione sequenziale assieme all’overhead di parallelizzazione.

dalle quali si ottengono e valutano i grafici:
* **Tempo vs core**: ci si aspetta una decrescita quasi iperbolica ($T(p) \approx T(1)/p$).
* **Serial fraction vs core**: dovrebbe rimanere costante per avere una buona stima di Amdahl, fondamentale per l'accuratezza delle analisi:
  + se variano molto con $p$ > instabilità data da overhead di parallelizzazione (sync, memoria, false sharing, cache misses, etc) *che aumentano col numero di core* > investire nel ridurre overhead/sbilanciamento.

* **Speedup vs core**: la curva reale dovrebbe avvicinarsi alla diagonale ideale ($S(p) = p$), ma, tenendo conto della frazione sequenziale $f$, l'andamento dovrebbe seguire la curva teorica di Amdahl usando la stima media di $f$ (vedi [Linear Fit](#linear-fit)):
  + Se vi è forte divergenza al variare dei core > overhead e/o sbilanciamento dei dati.
* **Efficienza strong vs core**: tipicamente cala oltre una certa soglia (limiti di Amdahl):
  + Se l’efficienza crolla presto > overhead di sincronizzazione o parte sequenziale troppo pesante.
  + Se aumenta bene fino a un certo numero di thread e poi si appiattisce > saturazione della memoria o sezioni sequenziali dominanti.

### Weak Scaling

Il **weak scaling** serve a capire se il programma può **gestire problemi sempre più grandi** su più risorse.

Viene scelta l'**unità di lavoro** $W_0$, per poi aumentare il numero di core $p$ e di conseguenza la dimensione del problema $W(p) = p \cdot W_0$, in modo tale che il numero di dati *per core* rimanga costante. L'unità di lavoro $W_0$ dev'essere sufficientemente grande da compensare l'overhead di parallelizzazione, e allo stesso tempo la memoria RAM totale richiesta sia sostenibile dal sistema per evitare swapping. 

Vengono calcolate le seguenti metriche: 
  * **Weak efficiency**: $E_{W_0}(p) = \frac{T(1)}{T(p)}$, dove $T$ è il tempo medio su più ripetizioni (per ridurre il rumore).
  * **Scaled speedup**: $S_{W_0}(p) = p \cdot E_{W_0}(p) = \frac{p \cdot T(1)}{T(p)}$, utile per validare la legge di Gustafson.
  * **Throughput**: $P_{W_0}(p) = \frac{W(p)}{T(p)} = \frac{p \cdot W_0}{T(p)}$ indica il lavoro eseguito per unità di tempo (Mpix/s). 

dalle quali si ottengono e valutano i seguenti grafici:
  * **Tempo vs core**: ideale è costante. Un aumento segnala overhead di comunicazione o memoria. Idealmente è costante perché ogni thread lavora sulla stessa unità di lavoro $W_0$:
      + Se cresce > overhead di comunicazione/sincronizzazione o saturazione di banda memoria.
      + Se decresce > qualche effetto collaterale positivo (cache locality, schedulazione più efficiente, ecc.), ma è raro e spesso sospetto.  
  * **Weak efficiency vs core**: ideale 100%. Accettabile ≥ 80–90% in HPC. Se l’efficienza cala, possibili cause sono:
      + comunicazione crescente
      + contesa sulle risorse (memoria, I/O)
      + overhead di sincronizzazione
      + load balancing
  * **Scaled speedup vs core**: Se rimane vicino alla diagonale $y = p$ significa che il programma scala bene. 
  * **Throughput vs core**: idealmente la produttività dovrebbe crescere linearmente col numero di thread, i.e. $P_{ideal}(p) = p \cdot P_{W_0}(1)$; quella reale tende ad appiattirsi a causa di overhead (sincronizzazioni, memory bottleneck, NUMA effects, etc), o addirittura peggiorare per saturazione delle risorse (e.g. bandwidth di memoria).

> :warning: **Warning**: weak scaling non è appropriato su problemi:
>   * di *Global reductions*: riduzioni globali dimostrano costi crescenti con p.
>   * il cui costo per unità lavoro cambia con costi non proporzionali (e.g. $O(n log n)$ ).

#### L'unità di lavoro per kip

La complessità del problema di Kernal Image Processing è $O(MNK^2)$, dove $M$ e $N$ sono le dimensioni dell'immagine su cui applicare la convoluzione, e $K$ quella del kernel quadrato.

Lo weak scaling richiedere di mantenere costante il lavoro per core, i.e. $MNK^2/p$, per cui l’unità naturale è il numero di pixel per core. Tuttavia, le dimensioni delle immagini a disposizione sono limitate (vedi [Images](https://github.com/marcopaglio/kip-sequential?tab=readme-ov-file#images)), e la scelta più semplice è di considerare un’unica dimensione (e.g. 4000x2000) e ripetere la stessa immagine più volte fino a raggiungere il totale richiesto, mantenendo a sua volta la dimensione del kernel costante; in particolare, si è scelto di *impilare* le immagini una sopra l'altra, cioè da $(M, N)$ si passa a $(M, pN)$.

> :bulb: **tip**: In alternativa, si potrebbe pensare di compensare la crescita dei core aumentando il kernel size invece della dimensione dell’immagine. Tuttavia, $K$ partecipa in proporzione quadratica alla complessità del problema per cui risulta difficile ottenere valori interi di $K$ che mantengono il lavoro per core costante. E.g: se per $p=1$ si utilizza $K_1=10$, per $p=2$ dovremmo scegliere $K_2=\sqrt{2} K_1=14.142...$ 

### Linear Fit

La stima media di $f$, con $0 \leq f \leq 1$, si può ottenere come regressione lineare dei $f(p)$, con $p > 1$, secondo il seguente procedimento:
  1. Legge di Amdahl: $S(p) = \frac{1}{f + (1-f)/p}$.
  2. Invertire: $\frac{1}{S(p)} = f + \frac{1}{p}(1-f)$ equivalente ad un'equazione lineare nella forma $y = a + b \cdot x$ con $y = \frac{1}{S(p)}$, $x = \frac{1}{p}$, $a = f$, e $b = 1 - f$.
  3. Una volta calcolati gli speedup $S(p)$, calcolare i relativi $y(p)$ e $x(p)$, dai quali ottenere una singola retta per regressione lineare di $y$ rispetto a $x$.
  4. Calcolare il punto di intersezione della retta con l'asse $y$, i.e. $a = f$.

Il valore così calcolato di $f$ è più robusto perchè utilizza tutti i punti e permette di ridurre il rumore della singola stima:
  * Se $f$ è alto > tanto sequenziale > redesign dell’algoritmo.
  * Se $f$ decresce dopo ottimizzazioni (meno sync, migliore bilanciamento, migliore locality) > si sta recuperando margine reale.
  * Se $f$ è basso ma lo speedup si appiattisce > possibile memory bandwidth > agire su locality, blocking, riduzione traffico memoria.




### Analisi dei grafici

La valutazione dei grafici dovrebbe essere utile per:
  * Capire se il programma scala quando aggiungi thread.
  * Capire dove collassa (NUMA, cache, memoria, sincronizzazione).
  * Quantificare i colli di bottiglia in termini di perdita di efficienza o throughput.

I grafici vanno letti congiuntamente per capire **se conviene scalare su più core, se serve un redesign, o se il programma è già vicino al massimo teorico**:
  * Se **strong scaling va male** > prima ottimizza, altrimenti il weak scaling non ha senso (ci saranno inefficienze ovunque).
  * Una volta che il codice scala “ragionevolmente” in strong scaling (cioè senza inefficienze banali) > esegui il weak scaling per valutare quanto bene la tua applicazione rimane performante quando cresce il problema.
  * Se **strong scaling è buono** ma **weak inefficiente**: l’algoritmo gestisce bene problemi fissi ma non cresce bene > il collo di bottiglia non è la parte sequenziale, ma la comunicazione e la memoria che crescono con la dimensione del problema.
  * Se **throughput scala bene** ma **tempo non cala**: l’algoritmo è adatto a problemi grandi, non a tempi ridotti.

Linee guida operative:
  1. Decidi quanti thread usare osservando il grafico Throughput+Efficienza:
      * I grafici di throughput e weak efficiency, presi congiuntamente, sono utili per decidere fino a che numero di thread vale la pena parallelizzare, in base a fino a quando i valori registrati rimangono vicini a quelli ideali: il punto in cui il throughput smette di crescere in modo proporzionale e l’efficienza crolla corrisponde al numero massimo di thread utile da sfruttare in parallelo.
      * Usa fino al punto in cui throughput continua a crescere e l’efficienza resta > ~0.7–0.8.
      * Oltre quel punto, più thread peggiorano solo il rapporto costi/benefici.
  2. Valida la scalabilità con lo Scaled Speedup:
      * Se segue bene la linea ideale → il problema è ben parallelizzabile, puoi pensare di scalare su più core/macchine.
      * Se cala molto → serve lavorare su riduzione degli overhead (meno comunicazioni, migliore suddivisione del lavoro).
Di conseguenza, Throughput+Efficienza → ti dice quanti thread ha senso usare; scaled Speedup → ti dice quanto bene scala davvero l’algoritmo.


> :pencil: **Note**: usare weak scaling in parallelo allo strong scaling, non come fase finale. Perché serve a capire se l’algoritmo rimane efficiente man mano che cresce il problema, non solo dopo aver “spremuto” lo strong scaling.

> :pencil: **Note**:Strong e weak scaling forniscono macro-indizi *se* vi sono possibili problematiche o miglioramenti; il profiling serve come micro-diagnosi per cercare *dove* sono le problematiche o i miglioramenti.

#### Core Fisici vs. virtuali

I **core fisici** sono unità di calcolo indipendenti, ciascuno dei quali può esporre 2 (o più) **core logici** (e.g. thread virtuali/HT/SMT); pertanto, questi ultimi condividono risorse hardware con il core fisico cui appartengono (pipeline, cache, ALU, unità di esecuzione).

Poiché i core logici non aumentano la potenza di calcolo, l'esito del loro utilizzo dipende dal workload:
  * Se è *latency-bound* o con tanti stall per cache miss > i thread logici possono aiutare a tenere occupata l’unità di calcolo, con aumento delle prestazioni fino al 20–30%.
  * Se è *compute-bound* e le risorse sono già sature > zero benefici, o addirittura peggioramenti dell’efficienza a causa di contenzione interna.

Lo scaling su thread logici può produrre risultati ingannevoli, per cui in genere conviene utilizzare solo i core fisici. Ciò non significa che non vadano usati, ma in tal caso i dati vanno interpretati adeguatamente.

Su OpenMP si può garantire la suddivisione dei thread su core differenti impostando le seguenti variabili d'ambiente:
```
OMP_PLACES=cores
OMP_PROC_BIND=TRUE
```
Qualora le richieste superino il massimo numero di core fisici disponibili, OpenMP mapperà i thread anche sui core virtuali.

### Quando fermarsi

Criteri pratici per fermarsi:
  * Marginal speedup: $\Delta{S}=S(p) - S(p/2)$, considerando di raddoppiare $p$ ad ogni step, scende sotto ~10–20% → poco valore ad aumentare p.
  * Efficienza: Un’efficienza sopra il 70% significa che l'utilizzo delle risorse è sufficientemente buono ed utile scalare su più thread o nodi; al di sotto, significa saturare le risorse e ulteriori thread non portano a benefici lineari.
  * Tempo: se $T(p)$ non migliora ($>~5%$) raddoppiando $p$, usa il $p$ più basso che dà lo stesso tempo.
  * Throughput: sopra 70-90% l’algoritmo scala sufficientemente bene. Al di sotto, l'overhead diviene evidente e conviene fermarsi.
