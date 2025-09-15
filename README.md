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
       
  3. **Analisi della scalabilità teorica**:
       * **Legge di Amdahl**: $S(p) = \frac{1}{f + \frac{1-f}{p}}$, per valutare il limite teorico dato dalla frazione sequenziale, mediante *strong scaling*.
       * **Legge di Gustafson**: valutare la bontà della parallelizzazione al crescere del problema, mediante *weak scaling*.
       
  4. **Profiling del codice parallelo** sui problemi che hanno generato comportamenti inattesi sugli scaling, col fine di determinarne le cause:
       
       |    Analysis       |      Cosa cercare       |        Possibili cause       |         Azioni consigliate              |
       |:-----------------:|:-----------------------:|:----------------------------:|:---------------------------------------:|
       | **Hotspot**       | sezioni dominanti       | codice sequenziale           | parallelizzare o ridisegnarre algoritmo |
       | **Threading**     | thread in idle          | Load imbalance               |             scheduling                  |
       |                   |                         | Sync overhead                | rimuovere/ridurre barriere              |
       | **Memory Access** | banda di memoria satura | Bottleneck                   | località/privatizzazione variabili      |

  6. **Analisi** del profiling ponderata ai risultati dello strong/weak scaling:
       * **Quantificare l’efficacia della parallelizzazione**: quanto spingersi con il numero di core.
       * **Individuare colli di bottiglia e problematiche**: memoria, load balance, sincronizzazioni, comunicazioni, etc.
       * **Guidare lo sviluppo**: capire se conviene lavorare sugli overhead, oppure ridisegnare l’algoritmo.
  
  7. **Ripetere** tuning > scaling > profiling > analysis finché non si raggiunge un compromesso accettabile.


### Strong Scaling

Lo **strong scaling** serve a capire se il programma può diventare più veloce.

Una volta fissata la dimensione del problema, all'aumentare del numero di core $p$ vengono calcolate le seguenti metriche:

  * **Speedup**: $S(p) = \frac{T(1)}{T(p)}$, dove $T(p)$ è il tempo misurato con $p$ core.
  * **Efficienza strong**: $E(p) = \frac{S(p)}{p}$.
  * **Serial fraction di Karp–Flatt**: $f(p) = \frac{\frac{1}{S(p)} - \frac{1}{p}}{1 - \frac{1}{p}}$, ottenuta dalla legge di Amdahl, misura la porzione sequenziale con aggiunto l’overhead di parallelizzazione (sincronizzazioni, imbalance, ecc.)

dalle quali si ottengono e valutano i grafici:

* **Tempo vs core**: ci si aspetta una decrescita quasi iperbolica ($T(p) \approx T(1)/p$).
* **Serial fraction vs core**:
  + Se $f(p)$ rimane circa costante, si ha una un'ottima stima di Amdahl, fondamentale per l'accuratezza complessiva delle analisi.
  + Se $f(p)$ cresce con $p$, si ha overhead/colli di bottiglia che aumentano coi thread (sync, memoria, false sharing, etc).
* **Speedup vs core**: la curva reale dovrebbe avvicinarsi alla diagonale ideale ($S(p) = p$), tuttavia, tenuto di conto della frazione sequenziale $f$, il confronto viene fatto con la curva della legge di Amdahl utilizzando $f$ generato da regressione lineare dei $f(p)$ (in modo da ridurre il rumore della singola stima). Se le curve distano molto al variare dei core, c'è un qualche problema di overhead/sbilanciamento dei dati.
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


I grafici vanno letti congiuntamente per capire **se conviene scalare su più core, se serve un redesign, o se il programma è già vicino al massimo teorico**: (TODO: questo si riferisce ad entrambi)

   * Se **throughput scala bene** ma **tempo non cala**: l’algoritmo è adatto a problemi grandi, non a tempi ridotti.
   * Se **strong scaling è buono** ma **weak inefficiente**: l’algoritmo gestisce bene problemi fissi ma non cresce bene.
   * Se entrambi sono scarsi: serve un redesign dell’algoritmo.



