# kip-parallel-OpenMP
Parallelized version of Kernel Image Processing, i.e. kernel filtering applied to 2D images, with OpenMP.


## Processo di parallelizzazione

Parallelizzare un programma sequenziale non significa solo “aggiungere thread”, ma richiede un percorso metodico per evitare false parallelizzazioni o addirittura rallentamenti, capire dove concentrare sforzi e analisi col fine di ottenere miglioramenti, e capire quanto bene l’algoritmo sia stato parallelizzato e quanto si possa ancora fare. A tal proposito, il workflow da seguire può essere così riassunto:

  1. **Profiling del codice sequenziale** per individuare dove intervenire:
       1. Identificare gli *hotspot*, i.e. funzioni che consumano più tempo.
       2. Capire se il collo di bottiglia è CPU, memoria, o I/O.
       
  2. **Inserire parallelismo incrementale** degli hotspot:
       * Riconoscere sezioni parallele
       * Riutilizzare i thread per più operazioni
       * Rimuovere le sincronizzazioni non necessarie
       * Privatizzare le variabili
       
  3. **Analisi della scalabilità teorica** fino a saturazione delle risorse:
       1. **Legge di Amdahl**: valutare il limite teorico dato dalla frazione sequenziale, mediante [*strong scaling*](#strong-scaling).
       2. **Legge di Gustafson**: valutare la bontà della parallelizzazione al crescere del problema, mediante [*weak scaling*](#weak-scaling).
       
  4. **Profiling del codice parallelo** mirato ai problemi che hanno generato comportamenti inattesi (teoria $\neq$ pratica) sugli scaling, col fine di determinarne le cause.
  
  5. **Ripetere** tuning > scaling > profiling finché non si raggiunge un compromesso accettabile.

---

### Strong Scaling

Lo **strong scaling** serve a capire quanto il programma beneficia della parallelizzazione, di fronte al fatto che la dimensione del problema non può crescere ma conta solo ridurre il tempo.

Una volta fissata la dimensione del problema, all'aumentare del numero di thread $p$ vengono calcolate le seguenti metriche:

  * **Wall-clock time**: $T(p)$ è il tempo medio su più ripetizioni (per ridurre il rumore):
      + Ci si aspetta una decrescita quasi iperbolica ($T(p) \approx \frac{T(1)}{p}$).
        
  * **Speedup**: $S(p) = \frac{T(1)}{T(p)}$:
      + La curva dovrebbe avvicinarsi alla diagonale ideale ($S(p) = p$), ma, tenendo conto della frazione sequenziale $f$, l'andamento dovrebbe seguire la curva teorica di Amdahl usando la stima di $f$ (vedi [linear fit](#linear-fit)).
      + Se vi è forte divergenza al variare dei thread > overhead di sincronizzazione e/o sbilanciamento dei dati.
        
  * **Efficienza strong**: $E(p) = \frac{S(p)}{p}$:
      + Tipicamente cala oltre una certa soglia (limiti di Amdahl).
      + Se l’efficienza crolla presto > overhead di sincronizzazione o parte sequenziale troppo pesante.
      + Se aumenta bene fino a un certo numero di thread e poi si appiattisce > saturazione della memoria o sezioni sequenziali dominanti.
        
  * **Karp–Flatt metric**: $f(p) = \frac{1/S(p) - 1/p}{1 - 1/p}$ misura la porzione sequenziale assieme all’overhead di parallelizzazione:
      + Dovrebbe rimanere costante per avere una buona stima di Amdahl, fondamentale per l'accuratezza delle analisi.
      + Se variano molto con $p$ > instabilità data dagli overhead di parallelizzazione (sync, memoria, false sharing, cache misses, etc) *che aumentano troppo col numero di thread*.


### Weak Scaling

Il **weak scaling** serve a capire se il programma può **gestire problemi sempre più grandi** su più risorse.

Viene scelta l'**unità di lavoro** $W_0$, per poi aumentare il numero di thread $p$ e di conseguenza la dimensione del problema $W(p) = p \cdot W_0$, in modo che il numero di dati *per thread* rimanga costante. $W_0$ dev'essere sufficientemente grande da compensare l'overhead di parallelizzazione, ma non troppo per evitare lo swapping della RAM (i.e. il sistema deve poter sostenere la memoria totale).

> :warning: **Warning**: weak scaling non è appropriato su problemi:
>   * Di *Global reductions*: riduzioni globali dimostrano costi crescenti con p.
>   * Il cui costo per unità lavoro cambia con costi non proporzionali (e.g. $O(n log n)$ ).

Vengono calcolate le seguenti metriche:

  * **Wall-clock time**: $T(p)$ è il tempo medio su più ripetizioni (per ridurre il rumore):
      + Idealmente è costante perché ogni thread lavora sulla stessa unità di lavoro $W_0$.
      + Se cresce > segnale di overhead di comunicazione/sincronizzazione o saturazione di banda di memoria.
      + Se decresce > qualche effetto collaterale positivo (cache locality, schedulazione più efficiente, ecc.), ma è raro e spesso sospetto.
        
  * **Weak efficiency**: $E_{W_0}(p) = \frac{T(1)}{T(p)}$:
      + Se rimane stabile vicino a 1 si ha un buon scaling.
      + Se cala > possibili cause: comunicazione crescente, contesa sulle risorse (memoria, I/O), overhead di sincronizzazione, load balancing.
        
  * **Scaled speedup**: $S_{W_0}(p) = p \cdot E_{W_0}(p) = \frac{p \cdot T(1)}{T(p)}$, utile per validare la legge di Gustafson:
      + Se rimane vicino alla diagonale $y = p$ significa che il programma scala bene.
      + Altrimenti serve lavorare sugli overhead (ridurre comunicazioni, migliore suddivisione del lavoro).
        
  * **Throughput**: $P_{W_0}(p) = \frac{W(p)}{T(p)} = \frac{p \cdot W_0}{T(p)}$ indica il lavoro eseguito per unità di tempo (Mpix/s):
      + Idealmente la produttività dovrebbe crescere linearmente col numero di thread, i.e. $P_{ideal}(p) = p \cdot P_{W_0}(1)$.
      + La curva reale tende ad appiattirsi a causa di overhead (sincronizzazioni, memory bottleneck, NUMA effects, etc), o addirittura peggiorare per saturazione delle risorse (e.g. bandwidth di memoria).


#### L'unità di lavoro per kip

La complessità del problema di Kernal Image Processing è $O(MNK^2)$, dove $M$ e $N$ sono le dimensioni dell'immagine su cui applicare la convoluzione, e $K$ quella del kernel quadrato.

Lo weak scaling richiedere di mantenere costante il lavoro per thread, i.e. $MNK^2/p$, per cui l’unità naturale è il numero di **pixel per thread**. Tuttavia, le dimensioni delle immagini a disposizione sono limitate (vedi [Images](https://github.com/marcopaglio/kip-sequential?tab=readme-ov-file#images)), e la scelta più semplice è di considerare un’unica dimensione (e.g. 4000x2000) e ripetere la stessa immagine più volte fino a raggiungere il totale richiesto, mantenendo a sua volta la dimensione del kernel costante; in particolare, si è scelto di *impilare* le immagini una sopra l'altra, cioè da $(M, N)$ si passa a $(M, pN)$.

> :bulb: **tip**: In alternativa, si potrebbe pensare di compensare la crescita dei thread aumentando il kernel size invece della dimensione dell’immagine. Tuttavia, $K$ partecipa in proporzione quadratica alla complessità del problema per cui risulta difficile ottenere valori interi di $K$ che mantengono il lavoro per thread costante. E.g: se per $p=1$ si utilizza $K_1=10$, per $p=2$ dovremmo scegliere $K_2=\sqrt{2} K_1=14.142...$ 

### Linear Fit

La stima media di $f$, con $0 \leq f \leq 1$, si può ottenere come regressione lineare delle metriche di Karp–Flatt $f(p)$, con $p > 1$:
  1. Si parte dalla legge di Amdahl: $S(p) = \frac{1}{f + (1-f)/p}$.
  2. Si inverte la formula: $\frac{1}{S(p)} = f + \frac{1}{p}(1-f)$, ottenendo un'equazione lineare nella forma $y = a + b \cdot x$ con $y = \frac{1}{S(p)}$, $x = \frac{1}{p}$, $a = f$, e $b = 1 - f$.
  3. Una volta calcolati gli speedup $S(p)$, si calcolano i relativi $y(p)$ e $x(p)$, dai quali si ottiene una singola retta per regressione lineare di $y$ rispetto a $x$.
  4. Si calcola il punto di intersezione della retta con l'asse $y$, i.e. $a = f$.

Il valore così calcolato di $f$ è più robusto perchè utilizza tutti i punti e permette di ridurre il rumore della singola stima:
  * Se $f$ è alto > sintomo di codice troppo sequenziale.
  * Se $f$ decresce dopo ottimizzazioni (meno sync, migliore bilanciamento, migliore locality) > si sta recuperando margine reale.
  * Se $f$ è basso ma lo speedup si appiattisce > possibile problema di memory bandwidth.

---

### Analisi dei dati

I dati ottenuti dagli scaling servono a:
  * **Quantificare l’efficacia della parallelizzazione** per decidere quanto spingersi con il numero di thread man mano che cresce il problema.
  * **Individuare colli di bottiglia e problematiche** laddove si hanno perdite di efficienza.
  * **Guidare lo sviluppo** e capire se conviene lavorare sugli overhead, ridisegnare l’algoritmo, oppure se il programma è già vicino al massimo teorico.

A tal proposito, strong e weak scaling devono essere utilizzati *in parallelo*:
  * Se **strong scaling va male** > prima si ottimizza, altrimenti il weak scaling non ha senso (ci saranno inefficienze ovunque).
  * Una volta che il codice scala “ragionevolmente” in strong scaling (cioè senza inefficienze banali) > si esegue il weak scaling per valutare l'applicazione al crescere del problema.

e i grafici vanno letti congiuntamente:
  * Se **strong scaling è buono** ma **weak inefficiente**: l’algoritmo gestisce bene problemi fissi ma non cresce bene > il collo di bottiglia non è la parte sequenziale, ma la comunicazione e la memoria che crescono con la dimensione del problema.
  * Se **throughput scala bene** ma **tempo non cala**: l’algoritmo è adatto a problemi grandi, non a tempi ridotti.
  * Il punto in cui il **throughput smette di crescere** in modo proporzionale e l’**efficienza crolla** corrisponde al numero massimo di thread da sfruttare in parallelo.

Strong e weak scaling forniscono macro-indizi *se* vi sono possibili problematiche o miglioramenti; dopodiché, il profiling serve come micro-diagnosi per cercare *dove* essi sono:
       
|    Analysis       |      Cosa cercare       |        Possibili cause       |     Azioni consigliate     |
|:-----------------:|:-----------------------:|:----------------------------:|:--------------------------:|
| **Hotspot**       | sezioni dominanti       | *codice sequenziale*         | parallelizzare             |
|                   |                         |                              | ridisegnarre algoritmo     |
|                   |                         |                              |                            |
| **Threading**     | thread in idle          | *Load imbalance*             | scheduling                 |
|                   |                         | *Sync overhead*              | rimuovere/ridurre barriere |
|                   |                         |                              |                            |
| **Memory Access** | banda di memoria satura | *Bottleneck memoria*         | migliorare locality        |
|                   | poor core utilization   |                              | tiling/blocking            |
|                   |                         |                              | migliorare uso cache       |
|                   |                         |                              | ridurre traffico memoria   |


#### Quando fermarsi

Se il numero di thread $p$ viene raddoppiato ad ogni step, la scelta di terminare il processo di parallelizzazione può considerare i seguenti criteri:
  * **Tempo** in *strong scaling*: se $T(p)$ non migliora di almeno $\sim$ 5-10%, usare il $p/2$, che dà circa lo stesso tempo, è la scelta migliore.
  * **Speedup** in *strong scaling*: se $\Delta{S}=S(p) - S(p/2)$ scende sotto $\sim$ 10–20% si ha poco valore ad aumentare p.
  * **Efficienza**: sopra il 70-80% i thread vengono sufficientemente usati; se al di sotto, significa saturare le risorse e ulteriori thread non portano a benefici lineari.
  * **Throughput**: finché continua a crescere e rimane sopra il 70-90% del throughput ideale, l’algoritmo scala sufficientemente bene; al di sotto, l'overhead diviene evidente e conviene fermarsi.

---

### Core Fisici vs. virtuali

I **core** sono unità di calcolo indipendenti, ciascuno dei quali può esporre di 2 (o più) **core logici**, o *thread*, i quali, pertanto, condividono risorse hardware (pipeline, cache, ALU, unità di esecuzione). Poiché i core logici non aumentano la potenza di calcolo, l'esito del loro utilizzo dipende dal workload:
  * Se è *latency-bound* o con tanti stall per cache miss > i thread possono aiutare a tenere occupata l’unità di calcolo, con aumento delle prestazioni fino al 20–30%.
  * Se è *compute-bound* e le risorse sono già sature > zero benefici, o addirittura peggioramenti dell’efficienza a causa di contenzione interna.

Lo scaling su core logici può produrre risultati ingannevoli, per cui in genere conviene utilizzare solo i core fisici. Ciò non significa che non vadano usati, ma in tal caso i dati vanno interpretati adeguatamente.

#### OpenMP

> :warning: **Warning**: Gli esperimenti dimostrano che forzare l'uso dei core fisici peggiora le prestazioni.

Su OpenMP si può garantire la suddivisione dei thread su core differenti impostando le variabili d'ambiente `OMP_PLACES=cores`e ` OMP_PROC_BIND=TRUE` prima di chiamare qualsiasi costrutto di OpenMP. Per esempio su C++:
```c++
#ifdef _WIN32
    _putenv_s("OMP_PROC_BIND", "TRUE");
    _putenv_s("OMP_PLACES",    "cores");
#elif __unix__
    setenv("OMP_PROC_BIND", "TRUE", 1);
    setenv("OMP_PLACES",   "cores", 1);
#endif
```
Qualora le richieste superino il massimo numero di core fisici disponibili, OpenMP mapperà i thread anche su core logici.
