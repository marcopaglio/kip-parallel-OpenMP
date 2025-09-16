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
       
  3. **Analisi della scalabilità teorica** al crescrere del numero di core *fisici* utilizzati fino a saturazione delle risorse:
       * **Legge di Amdahl**: valutare il limite teorico dato dalla frazione sequenziale, mediante [*strong scaling*](#strong-scaling).
       * **Legge di Gustafson**: valutare la bontà della parallelizzazione al crescere del problema, mediante [*weak scaling*](#weak-scaling).
       
  4. **Profiling del codice parallelo** applicato ai problemi che hanno generato comportamenti inattesi (teoria $\neq$ pratica) sugli scaling, col fine di:
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
  
  7. **Ripetere** tuning > scaling > profiling finché non si raggiunge un compromesso accettabile.


### Strong Scaling

Lo **strong scaling** serve a capire se il programma può diventare più veloce.

Una volta fissata la dimensione del problema, all'aumentare del numero di core $p$ vengono calcolate le seguenti metriche:
  * **Speedup**: $S(p) = \frac{T(1)}{T(p)}$, dove $T(p)$ è il tempo misurato con $p$ core.
  * **Efficienza strong**: $E(p) = \frac{S(p)}{p}$.
  * **Serial fraction di Karp–Flatt**: $f(p) = \frac{1/S(p) - 1/p}{1 - 1/p}$, ottenuta dalla legge di Amdahl, misura la porzione sequenziale assieme all’overhead di parallelizzazione (sincronizzazioni, imbalance, ecc.)

dalle quali si ottengono e valutano i grafici:
* **Tempo vs core**: ci si aspetta una decrescita quasi iperbolica ($T(p) \approx T(1)/p$).
* **Serial fraction vs core**: dovrebbe rimanere circa costanti per avere una buona stima di Amdahl, fondamentale per l'accuratezza delle analisi:
  + se variano molto con $p$ > l'instabilità è data da colli di bottiglia che non sono il puro “sequenziale”, ma overhead di parallelizzazione (sync, memoria, false sharing, cache misses, etc) *che aumentano col numero di core* > investire nel ridurre overhead/sbilanciamento.

* **Speedup vs core**: la curva reale dovrebbe avvicinarsi alla diagonale ideale ($S(p) = p$), ma, tenendo conto della frazione sequenziale $f$, l'andamento dovrebbe seguire la curva teorica di Amdahl usando la stima media di $f$ (vedi [Linear Fit](#linear-fit)):
  + Se vi è forte divergenza al variare dei core > problema di overhead e/o sbilanciamento dei dati.
* **Efficienza strong vs core**: tipicamente cala oltre una certa soglia (limiti di Amdahl):
  + Se l’efficienza crolla presto > overhead di sincronizzazione o parte sequenziale troppo pesante.
  + Se aumenta bene fino a un certo numero di thread e poi si appiattisce > saturazione della memoria o sezioni sequenziali dominanti.

### Weak Scaling

Il **weak scaling** serve a capire se il programma può **gestire problemi sempre più grandi** su più risorse.

Aumentando il numero di core, si aumenta anche il numero di dati in modo tale che la dimensione del problema *per core* rimanga costante, e vengono calcolate le seguenti metriche: 
  * **Weak efficiency**: $E_w(p) = \frac{T(1)}{T(p)}$ (tempo atteso costante).
  * **Scaled speedup**: $S_w(p) = \frac{p \cdot T(1)}{T(p)}$.
  * **Throughput**: lavoro eseguito per unità di tempo (Mpix/s, FLOP/s…).

dalle quali si ottengono e valutano i seguenti grafici:
  * **Tempo vs core**: ideale è costante. Un aumento segnala overhead di comunicazione o memoria.
  * **Weak efficiency vs core**: ideale 100%. Accettabile ≥ 80–90% in HPC. Se l’efficienza cala ⇒ colli di bottiglia in memoria, comunicazioni, o load balancing.
  * **Scaled speedup vs core**: dovrebbe seguire la diagonale $y = p$.
  * **Throughput vs core**: ideale cresce linearmente ($p \cdot \text{Throughput}_1$), quello reale tende a saturarsi.


### Linear Fit

La stima media di $f$, con $0 \leq f \leq 1$, si può ottenere come regressione lineare dei $f(p)$, con $p > 1$, secondo il seguente procedimento:
  1. Legge di Amdahl: $$S(p) = \frac{1}{f + (1-f)/p}$$
  2. Invertire: $$\frac{1}{S(p)} = f + \frac{1}{p}(1-f)$$
  3. Sin ha un'equazione lineare $$y = a + b \cdot x$$ con $y = \frac{1}{S(p)}$, $x = \frac{1}{p}$, $a = f$, e $b = 1 - f$
  4. Una volta calcolati gli $f(p)$, si calcolano i relativi $y(p)$ e $x(p)$, dai quali si ottiene una singola retta per regressione lineare di $y$ rispetto a $x$
  5. Si calcola il punto di intersezione della retta con l'asse $y$, e si ottiene $a = f$

Il valore così calcolato di $f$ è più robusto perchè utilizza tutti i punti e permette di ridurre il rumore della singola stima:
  * Se $f$ è alto > tanto sequenziale > redesign dell’algoritmo.
  * Se $f$ decresce dopo ottimizzazioni (meno sync, migliore bilanciamento, migliore locality) > si sta recuperando margine reale.
  * Se $f$ è basso ma lo speedup si appiattisce > possibile memory bandwidth > agire su locality, blocking, riduzione traffico memoria.




### Quando fermarsi

I grafici vanno letti congiuntamente per capire **se conviene scalare su più core, se serve un redesign, o se il programma è già vicino al massimo teorico**:

   * Se **throughput scala bene** ma **tempo non cala**: l’algoritmo è adatto a problemi grandi, non a tempi ridotti.
   * Se **strong scaling è buono** ma **weak inefficiente**: l’algoritmo gestisce bene problemi fissi ma non cresce bene.
   * Se entrambi sono scarsi: serve un redesign dell’algoritmo.

Amdahl: utile per capire se la parallelizzazione è “sufficiente” rispetto al problema dato:
  * Seleziono un problema di dimensione fissa e valuto la curva di strong scaling fino a saturazione delle risorse (aumentando il numero di core utilizzati)
Gustafson: utile per capire se il programma resta efficiente su problemi grandi (tipico in HPC):
  * Una volta che il codice scala “ragionevolmente” in strong scaling (cioè senza inefficienze banali), esegui il weak scaling per valutare quanto bene la tua applicazione rimane performante quando cresce il problema.
  * Se strong scaling va male → prima ottimizza, altrimenti il weak scaling non ha senso (avrai inefficienze ovunque).
  * Se strong scaling va bene, ma il weak scaling mostra caduta di efficienza → il collo di bottiglia non è la parte sequenziale, ma la comunicazione e la memoria che crescono con la dimensione del problema.
  * usa weak scaling in parallelo allo strong scaling, non come fase finale. Perché serve a capire se l’algoritmo rimane efficiente man mano che cresce il problema, non solo dopo aver “spremuto” lo strong scaling.

È lo step in cui verifichi se la tua parallelizzazione rimane utile su grandi macchine o se l’aumento di memoria/comunicazioni distrugge l’efficienza.
Strong e weak scaling forniscono macro-indizi *se* vi sono possibili problematiche o miglioramenti; il profiling serve come micro-diagnosi per cercare *dove* sono le problematiche o i miglioramenti.

Criteri pratici per fermarsi:
  * Marginal speedup: $\Delta{S}=S(p) - S(p/2)$, considerando di raddoppiare $p$ ad ogni step, scende sotto ~10–20% → poco valore ad aumentare p.
  * Efficienza: quando $E(p) < 50%$
  * Tempo: se $T(p)$ non migliora ($>~5%$) raddoppiando $p$, usa il $p$ più basso che dà lo stesso tempo.
