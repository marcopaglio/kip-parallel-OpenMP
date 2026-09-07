## Kip-parallel-OpenMP - SoA

This is the *Structure-of-Arrays* version of **kip-parallel-OpenMP**. An alternative versions can be found at the [AoS folder](../AoS "AoS version of kip-parallel-OpenMP").

### Parallelization iter

Il processo di parallelizzazione ha coinvolto unicamente il metodo `ImageProcessing::convolution` e ha seguito un percorso abbastanza coerente con i risultati provvisori via via ottenuti, così riassumibile: anzitutto è stata applicata la parallelizzazione OpenMP del ciclo sulle righe, dopodiché è stato esplorato lo scheduling e la privatizzazione delle variabili, poi `collapse` e le differenze con la vettorizzazione del kernel, fino alla suddivisione manuale del lavoro.

#### Parallelizzazione del ciclo esterno e confronto degli scheduler

La prima versione parallela naturale è stata quella in cui OpenMP distribuisce le iterazioni del ciclo esterno su `y`, i.e. lungo le righe dell'immagine:

```cpp
#pragma omp parallel for schedule(...)
for (unsigned int y = 0; y < outputHeight; ++y) {
    for (unsigned int x = 0; x < outputWidth; ++x) {
        ...
    }
}
```

L'ipotesi iniziale era che `schedule(static)` fosse la scelta più adatta. Dal punto di vista algoritmico, infatti, ogni iterazione di `y` esegue essenzialmente lo stesso lavoro: percorre `outputWidth` pixel e per ciascun pixel esegue una convoluzione di `order × order` elementi. `static` avrebbe quindi dovuto fornire un buon bilanciamento con un overhead inferiore a `dynamic` e `guided`.

Le misure hanno però mostrato qualcosa di diverso. Già nei primi esperimenti, si è osservato che **`dynamic` e `guided` erano più veloci di `static`**, con un vantaggio di circa il 10%; questo risultato consiglia che l'*uniformità del numero di operazioni non si traduce necessariamente in un tempo di esecuzione perfettamente uniforme fra i thread*. 

Ulteriore prova a supporto di questa impressione è stata ottenuta dal confronto fra **`static,1` e `static`**, i quali **hanno fornito prestazioni pressoché identiche**. Il fatto che `static,1` non migliori sensibilmente rispetto allo `static` standard è interessante perché i due distribuiscono le righe in maniera molto diversa: il primo in modo ciclico, il secondo normalmente per blocchi contigui. Questo rende meno convincente l'ipotesi che il vantaggio di `dynamic` dipenda principalmente dalla particolare posizione delle righe assegnate ai thread. Il risultato punta maggiormente verso la capacità di `dynamic` di assorbire piccole differenze nei tempi effettivi dei thread.

Infine, si è provato anche a controllare manualmente la dimensione dei chunk, usando valori derivati da `outputHeight / (omp_get_num_threads() * min_chunk)` con `min_chunk` pari a 1, 2, 4 e 8. In questo caso **le dimensioni di blocco impostate esplicitamente hanno dato risultati peggiori rispetto alla dimensione scelta di default dai rispettivi scheduler**. Di conseguenza, la configurazione che si è consolidata come migliore è rimasta:

```cpp
#pragma omp parallel for schedule(dynamic)
```

#### Introduzione di `collapse`

Successivamente si è provato a sfruttare l'indipendenza completa dei primi due cicli, i.e. lo spostamento lungo le righe (`y`) e le colonne (`x`) dell'immagine, utilizzando `collapse(2)` con l'obiettivo di aumentare lo spazio di iterazioni disponibile al runtime OpenMP. Invece di distribuire soltanto `outputHeight` righe, OpenMP avrebbe potuto distribuire l'intero spazio `outputHeight × outputWidth` dei pixel.

In generale, però, si è osservato **un leggero peggioramento delle prestazioni**. Questo è motivabile con il fatto che il parallelismo applicato sul solo ciclo `y` fosse già ampiamente sufficiente data la dimensione delle immagini (> 4000x2000 pixel), e pertanto l'introduzione di `collapse` non risolve una carenza di lavoro da distribuire.

#### Scelta tra `shared` e `firstprivate`

Un'altra parte della sperimentazione ha riguardato la classificazione delle variabili nella regione OpenMP, resa esplicita tramite `default(none)`. La configurazione attuale è:

```cpp
shared(reds, greens, blues, originalReds, originalGreens, originalBlues, outputHeight) \
firstprivate(width, outputWidth, order, kernelWeights)
```

L'idea generale seguita è stata distinguere tra variabili grandi o condivise naturalmente tra tutti i thread e variabili scalari, piccole e molto frequentemente utilizzate; per queste ultime, `firstprivate` può essere ragionevole perché ogni thread riceve una propria copia inizializzata con il valore originale, mentre per strutture dati più grandi, invece, `shared` evita il costo della copia privata. A tal proposito:

- Per i vettori di output (`reds`, `greens`, `blues`) la scelta `shared` è naturale e necessaria. Tutti i thread devono contribuire allo stesso risultato finale, ma ciascuna iterazione `(y,x)` scrive in una posizione differente, per cui non vi possono essere race condition.
- Lo stesso ragionamento, ma per motivi diversi, vale per `originalReds`, `originalGreens`, `originalBlues`. Questi vettori sono utilizzati esclusivamente in lettura e sono molto grandi rispetto alle altre variabili della regione parallela. `shared` è quindi la scelta più sensata: non esiste alcun rischio di data race dovuto alle letture concorrenti e una copia per thread avrebbe un costo potenzialmente molto elevato in memoria e inizializzazione.
- `outputHeight` è stato mantenuto `shared`. È un valore scalare read-only e viene utilizzato principalmente come limite dello spazio di iterazione esterno. Dal punto di vista correttezza potrebbe essere anche `firstprivate`, ma essendo poco utilizzato all'interno del kernel vero e proprio non c'è una motivazione concreta per crearne una copia privata per thread.
- Per `width`, `outputWidth` e `order`è stato invece scelto `firstprivate`. La motivazione è che si tratta di scalari molto piccoli, quindi il costo di duplicazione è trascurabile; inoltre vengono utilizzati frequentemente nella regione parallela. 
- Il caso più ambiguo è stato `kernelWeights`: di fatto, non è uno scalare ma si tratta di un contenitore di $order^2$ coefficienti. Poiché read-only, si potrebbe pensare che `shared` sia preferibile per `kernelWeights`, in modo da evitarsi anche il costo di costruzione e copia del vettore per ogni thread. Tuttavia i risultati sperimentali mostrano piccole variazioni dipendenti dalla dimensione del kernel: la conclusione più corretta è che non vi sia un vincitore assoluto, ma di continuare a tenerla in considerazione per i futuri esperimenti. Di fatto, tali differenze sembrano riconducibili più a effetti microarchitetturali che a una differenza fondamentale nel modello OpenMP. Per il momento, si è optato per mantenerla privata per ogni thread, seguendo l'idea di privatizzare quanto più possibile.

#### Riduzione SIMD sul kernel

Un'altra fetta di esperimenti ha riguardato i due cicli interni del kernel, i.e. lungo le righe (`j`) e le colonne (`i`) del kernel:

```cpp
for (unsigned int j = 0; j < order; ++j) {
    for (unsigned int i = 0; i < order; ++i) {
        ...
    }
}
```

Poiché questi cicli accumulano i valori di `channelRed`, `channelGreen` e `channelBlue`, si prestano particolarmente bene per l'applicazione del pattern di *riduzione*. Affinché potesse essere applicato senza dover generare ulteriori thread mediante un’altra direttiva `for`, oltre a quella già applicata al ciclo `y` — approccio che, sperimentalmente, ha mostrato una drastica riduzione delle prestazioni — è stato considerato l’utilizzo della direttiva `simd`. A differenza della precedente, questa direttiva istruisce il compilatore OpenMP a vettorizzare il ciclo successivo senza ricorrere al worksharing.
Le modalità testate sono state principalmente due:

1. Tramite l'utilizzo combinato di `collapse`:

```cpp
#pragma omp simd collapse(2) reduction(+:channelRed, channelGreen, channelBlue)
for (unsigned int j = 0; j < order; ++j) {
    for (unsigned int i = 0; i < order; ++i) {
        ...
    }
}
```

Tuttavia, **le prestazioni sono peggiorate in modo consistente e il peggioramento è aumentato al crescere delle dimensioni degli input**.

2) Applicando la SIMD direttamente a `i`, cioè agli elementi consecutivi di una stessa riga dell'immagine e del kernel:

```cpp
for (unsigned int j = 0; j < order; ++j) {
    const unsigned int posBase = (y + j) * width + x;
    const unsigned int kwBase = j * order;
#pragma omp simd reduction(+:channelRed, channelGreen, channelBlue)
    for (unsigned int i = 0; i < order; ++i) {
        ...
    }
}
```

**Questa forma ha ottenuto miglioramenti** veramente consistenti, **di oltre 2x per i kernel più grossi**. A tal proposito, si è voluto etichettare tale versione parallela come *paralellizzazione #1*, cioè la migliore di questa prima parte di esperimenti, i cui risultati sono riportati nell'analoga sezione dei risultati sperimentali. 

Un'altra piccola ottimizzazione che si può notare è quella di aver portato fuori dal ciclo `i` due calcoli invarianti di `posBase` e `kwBase` riducendo l'aritmetica degli indici nel ciclo più interno.

#### Divisione manuale del lavoro tra thread

Il passo successivo è stato quello di provare ad eliminare completamente la worksharing construct `omp for`, mantenendo la sola regione `parallel`, e calcolando manualmente per ogni thread un intervallo di righe.

La prima versione è stata:

```cpp
lowerBound = outputHeight / nthreads * thread_id;

upperBound =
    thread_id == nthreads - 1
    ? outputHeight
    : outputHeight / nthreads * (thread_id + 1);
```

L'obiettivo era verificare se eliminando l'intervento dello scheduler OpenMP fosse possibile ottenere prestazioni migliori. Il risultato è stato però **identico o minimamente peggiore**.

È stato quindi individuato un problema nella prima suddivisione: tutto il resto della divisione veniva assegnato all'ultimo thread, il che avrebbe potuto rendere la suddivisione manuale vana dal momento che tutti i thread avrebbero dovuto attendere la fine del maggior carico di lavoro dell'ultimo thread. Si è dunque proposto una modalità *fair*, in cui le righe residue vengono distribuite una per volta ai primi thread:

```cpp
const unsigned int base = outputHeight / nthreads;
const unsigned int remainder = outputHeight % nthreads;

lowerBound =
    thread_id * base +
    std::min<unsigned int>(thread_id, remainder);

upperBound =
    lowerBound + base +
    (thread_id < remainder ? 1 : 0);
```

Questa modifica **ha prodotto piccoli miglioramenti in tutti gli esperimenti rispetto alla prima divisione manuale**, confermando che il precedente squilibrio fosse reale. Tuttavia, anche dopo aver reso la partizione quasi perfettamente uniforme, **nel complesso la suddivisione manuale non ha portato miglioramenti**. Questo è probabilmente uno dei risultati più significativi ottenuti finora. Dimostra che il vantaggio del dynamic scheduling non deriva semplicemente da un'implementazione inefficiente dello static scheduling di OpenMP.

### Experimental Results

TODO

### Profiling Results

TODO
