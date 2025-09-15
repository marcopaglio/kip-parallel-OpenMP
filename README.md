# kip-parallel-OpenMP
Parallelized version of Kernel Image Processing, i.e. kernel filtering applied to 2D images, with OpenMP.







## Processo di parallelizzazione

Parallelizzare un programma sequenziale non significa solo “aggiungere thread”, ma richiede un percorso metodico per evitare false parallelizzazioni, o addirittura rallentamenti, capire dove concentrare sforzi e analisi col fine di ottenere miglioramenti, e capire **quanto bene l’algoritmo** sia stato parallelizzato e quanto si possa ancora fare. A tal proposito, il workflow seguito in questo lavoro può essere così riassunto:

  1. **Profiling del codice sequenziale** (e.g. tramite VTune) per individuare dove intervenire:
     
       * Identificare gli *hotspot*, i.e. funzioni che consumano più tempo.
       * Capire se il collo di bottiglia è CPU, memoria, o I/O.
       
  2. **Parallelizzazione incrementale**:
     
       * Riconoscere sezioni parallele
       * Privatizzare le variabili
       * Scegliere scheduling
       
  3. **Analisi della scalabilità teorica**:
     
       * **Legge di Amdahl**: valutare il limite massimo dato dalla frazione sequenziale.
       * **Legge di Gustafson**: valutare la bontà della parallelizzazione al crescere del problema.
       
  4. **Profiling del codice parallelo**:
     
       * Localizzare overhead
       * Riconoscere colli di bottiglia (memoria, load balance, etc)

     La profilazione del codice parallelo deve essere integrata e ponderata in base ai risultati ottenuti da: (TODO: mettere qui dettagli?)
     
       * Lo **strong scaling**: verificare riduzione del tempo di esecuzione fissando il problema e aumentando il numero di core.
       * Il **weak scaling**: misurare la variazione di tempo quando la dimensione del problema cresce proporzionalmente al numero di core.

  5. **Analisi dei grafici**:

       * **Quantificare l’efficacia della parallelizzazione** (quanto conviene spingersi con il numero di core).
       * **Individuare colli di bottiglia** (memoria, sincronizzazioni, comunicazioni).
       * **Guidare lo sviluppo**: capire se conviene lavorare su bilanciamento del carico, comunicazioni, o ridisegnare l’algoritmo.
  
  6. **Iterazione**: ripetere profiling → tuning → scaling finché non si raggiunge un compromesso accettabile.


### Strong Scaling

Lo **strong scaling** serve a capire se il programma può diventare più veloce.

All'aumentare del numero di core, e mantenendo il problema fisso, vengono calcolate due metriche:

  * **Speedup**: $S(p) = \frac{T(1)}{T(p)}$.
  * **Efficienza strong**: $E(p) = \frac{S(p)}{p}$.

dalle quali si ottengono e valutano tre tipologie di grafici:

* **Tempo vs core**: ci si aspetta una decrescita quasi iperbolica ($T(p) \approx T(1)/p$).
* **Speedup vs core**: la curva reale dovrebbe avvicinarsi alla diagonale ideale ($S(p) = p$).
* **Efficienza strong vs core**: tipicamente cala oltre una certa soglia (limiti di Amdahl). Se l’efficienza crolla presto ⇒ overhead di sincronizzazione o parte sequenziale troppo pesante.


### Weak Scaling

Il **weak scaling** serve a capire se il programma può **gestire problemi sempre più grandi** su più risorse.

Aumentando il numero di core, si aumenta anche il numero di dati in modo tale che la dimensione del *problema per core* rimanga costante, e vengono calcolate le seguenti metriche: 

  * **Weak efficiency**: $E_w(p) = \frac{T(1)}{T(p)}$ (tempo atteso costante).
  * **Scaled speedup**: $S_w(p) = \frac{p \cdot T(1)}{T(p)}$.
  * **Throughput**: lavoro eseguito per unità di tempo (Mpix/s, FLOP/s…).

dalle quali si ottengono e valutano quattro tipologie di grafici:

  * **Tempo vs core**: ideale è costante. Un aumento segnala overhead di comunicazione o memoria.
  * **Weak efficiency**: ideale 100%. Accettabile ≥ 80–90% in HPC. Se l’efficienza cala ⇒ colli di bottiglia in memoria, comunicazioni, o load balancing.
  * **Scaled speedup vs core**: dovrebbe seguire la diagonale $y = p$.
  * **Throughput**: ideale cresce linearmente ($p \cdot \text{Throughput}_1$), quello reale tende a saturarsi.


I grafici vanno letti congiuntamente per capire **se conviene scalare su più core, se serve un redesign, o se il programma è già vicino al massimo teorico**: (TODO: questo si riferisce ad entrambi)

   * Se **throughput scala bene** ma **tempo non cala**: l’algoritmo è adatto a problemi grandi, non a tempi ridotti.
   * Se **strong scaling è buono** ma **weak inefficiente**: l’algoritmo gestisce bene problemi fissi ma non cresce bene.
   * Se entrambi sono scarsi: serve un redesign dell’algoritmo.



