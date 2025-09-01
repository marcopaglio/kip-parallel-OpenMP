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
   * **Legge di Amdahl**: limite massimo dato dalla frazione sequenziale.
   * **Legge di Gustafson**: crescita del problema rende più favorevole il parallelismo.
4. **Profiling del codice parallelo**:
   * Localizzare overhead
   * Riconoscere colli di bottiglia (memoria, load balance, etc)

In particolare, la profilazione deve essere integrata con: 
* Lo **strong scaling**: verificare riduzione del tempo di esecuzione fissando il problema e aumentando i core.
* Il **weak scaling**: misura come varia il tempo quando la dimensione del problema cresce proporzionalmente ai core.

Queste analisi sono fondamentali perché consentono di:
1. **Quantificare l’efficacia della parallelizzazione** (dove conviene spingersi con il numero di core).
2. **Individuare colli di bottiglia** (memoria, sincronizzazioni, comunicazioni).
3. **Guidare lo sviluppo**: capire se conviene lavorare su bilanciamento del carico, comunicazioni, o ridisegnare l’algoritmo.

### Strong Scaling

#### Definizione

* Problema fissato, core crescenti.
* Metriche principali:

  * **Speedup**: $S(p) = \frac{T(1)}{T(p)}$.
  * **Efficienza strong**: $E(p) = \frac{S(p)}{p}$.

#### Interpretazione dei grafici

* **Tempo vs core**: ci si aspetta una decrescita quasi iperbolica ($T(p) \approx T(1)/p$).
* **Speedup vs core**: la curva reale dovrebbe avvicinarsi alla diagonale ideale ($S(p) = p$).
* **Efficienza strong vs core**: tipicamente cala oltre una certa soglia (limiti di Amdahl).

#### Utilità nel processo

* Serve a capire **quanto un problema fisso beneficia della parallelizzazione**.
* È cruciale per applicazioni interattive o real-time, dove la dimensione del problema non può crescere ma conta solo ridurre il tempo.

### Weak Scaling

#### Definizione

* Problema per core costante: raddoppio i core, raddoppio i dati.
* Metriche principali:

  * **Weak efficiency**: $E_w(p) = \frac{T(1)}{T(p)}$ (tempo atteso costante).
  * **Scaled speedup**: $S_w(p) = \frac{p \cdot T(1)}{T(p)}$.
  * **Throughput**: lavoro eseguito per unità di tempo (Mpix/s, FLOP/s…).

#### Interpretazione dei grafici

* **Tempo vs core**: ideale è costante. Un aumento segnala overhead di comunicazione o memoria.
* **Weak efficiency**: ideale 100%. Accettabile ≥ 80–90% in HPC.
* **Scaled speedup vs core**: dovrebbe seguire la diagonale $y = p$.
* **Throughput**: ideale cresce linearmente ($p \cdot \text{Throughput}_1$), quello reale tende a saturarsi.

#### Utilità nel processo

* Serve a capire se il programma può **gestire problemi sempre più grandi** su più risorse.
* È cruciale per simulazioni scientifiche, big data, deep learning.

   * Se l’efficienza crolla presto ⇒ overhead di sincronizzazione o parte sequenziale troppo pesante.
4. **Weak scaling**: verificare se l’app regge problemi crescenti.
 Decidere il giusto numero di thread da utilizzare
   * Se l’efficienza cala ⇒ colli di bottiglia in memoria, comunicazioni, o load balancing.
5. **Analisi dei grafici**:

   * Se **throughput scala bene** ma **tempo non cala**, l’algoritmo è adatto a problemi grandi, non a tempi ridotti.
   * Se **strong scaling è buono** ma **weak inefficiente**, l’algoritmo gestisce bene problemi fissi ma non cresce bene.
   * Se entrambi sono scarsi ⇒ serve un redesign dell’algoritmo.
6. **Iterazione**: ripetere profiling → tuning → scaling finché non si raggiunge un compromesso accettabile.










# 🔹 Interpretazione congiunta dei grafici

Per non avere ridondanze:

* **Tempo medio** (strong) + **Throughput** (weak) danno la visione pratica: velocità e produttività.
* **Efficienza** (sia strong che weak) è utile per diagnosticare overhead nascosti.
* **Speedup** è più “didattico” ma meno utile se già hai efficienza.

👉 Quindi, in un report finale:

* tieni **Tempo vs core** e **Throughput vs core** come indicatori principali;
* usa **Efficienza** per discutere criticità;
* puoi omettere speedup se serve concisione.

---

# 🔹 Conclusione

Un processo di parallelizzazione ottimale richiede **profiling accurato + analisi di strong/weak scaling**.

* Lo **strong scaling** ti dice se il programma può diventare più veloce.
* Il **weak scaling** ti dice se il programma può affrontare problemi più grandi.
* I grafici (tempo, efficienza, throughput) vanno letti congiuntamente per capire **se conviene scalare su più core, se serve un redesign, o se il programma è già vicino al massimo teorico**.


