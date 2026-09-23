## Kip-parallel-OpenMP - SoA

This is the *Structure-of-Arrays* version of **kip-parallel-OpenMP**. Alternative versions can be found at:

- [AoS folder](../AoS "AoS version of kip-parallel-OpenMP")
- [Parallel#0 branch](https://github.com/marcopaglio/kip-sequential/tree/parallel%230/SoA "First parallelization of kip-parallel-openMP's SoA version")
- [Parallel#1 branch](https://github.com/marcopaglio/kip-sequential/tree/parallel%231/SoA "Second parallelization of kip-parallel-openMP's SoA version")

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

L'ipotesi iniziale era che `schedule(static)` fosse la scelta più adatta. Dal punto di vista algoritmico, infatti, ogni iterazione di `y` esegue essenzialmente lo stesso lavoro: percorre `outputWidth` pixel e per ciascun pixel esegue una convoluzione di `order × order` elementi. `static` avrebbe quindi dovuto fornire un buon bilanciamento con un overhead inferiore a `dynamic` e `guided`. Le misurazioni hanno però mostrato qualcosa di diverso:

- Già nei primi esperimenti, si è osservato che **`dynamic` e `guided` erano più veloci di `static`**, con un vantaggio di circa il 10%; questo risultato consiglia che l'*uniformità del numero di operazioni non si traduce necessariamente in un tempo di esecuzione perfettamente uniforme fra i thread*. 
- Ulteriore prova a supporto di questa impressione è stata ottenuta dal confronto fra **`static,1` e `static`**, i quali **hanno fornito prestazioni pressoché identiche**. Il fatto che `static,1` non migliori sensibilmente rispetto allo `static` standard è interessante perché i due distribuiscono le righe in maniera molto diversa: il primo in modo ciclico, il secondo normalmente per blocchi contigui. Questo rende meno convincente l'ipotesi che il vantaggio di `dynamic` dipenda principalmente dalla particolare posizione delle righe assegnate ai thread. Il risultato punta maggiormente verso la capacità di `dynamic` di assorbire piccole differenze nei tempi effettivi dei thread.

Infine, si è provato anche a controllare manualmente la dimensione dei chunk, usando valori derivati da `outputHeight / (omp_get_num_threads() * min_chunk)` con `min_chunk` pari a 1, 2, 4 e 8. In questo caso **le dimensioni di blocco impostate esplicitamente hanno dato risultati peggiori rispetto alla dimensione scelta di default dai rispettivi scheduler**. Di conseguenza, la configurazione che si è consolidata come migliore è rimasta:

```cpp
#pragma omp parallel for schedule(dynamic)
```

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
- Il caso più ambiguo è stato `kernelWeights`: di fatto, non è uno scalare ma si tratta di un contenitore di $order^2$ coefficienti. Poiché read-only, si potrebbe pensare che `shared` sia preferibile per `kernelWeights`, in modo da evitarsi anche il costo di costruzione e copia del vettore per ogni thread. Tuttavia i risultati sperimentali mostrano piccole variazioni dipendenti dalla dimensione del kernel: la conclusione più corretta è che non vi sia un vincitore assoluto, ma di continuare a tenerla in considerazione per i futuri esperimenti. Di fatto, tali differenze sembrano riconducibili più a effetti microarchitetturali che a una differenza fondamentale nel modello OpenMP. In conclusione, per il momento, si è optato per mantenerla privata per ogni thread seguendo l'idea di privatizzare quanto più possibile.

Complessivamente, queste prime due modifiche hanno determinato degli ottimi risultati rispetto alla versione sequenziale del problema, per cui si identifica con *Parallel#0* la versione del programma come delineata finora. I relativi risultati sulle performance sono riportati in [Parallel#0's Experimental Results](https://github.com/marcopaglio/kip-parallel-OpenMP/tree/parallel%230/SoA#experimental-results "Experimental results section for the first parallelization of kip-parallel-openMP's SoA version").

#### Introduzione di `collapse`

Successivamente si è provato a sfruttare l'indipendenza completa dei primi due cicli, i.e. lo spostamento lungo le righe (`y`) e le colonne (`x`) dell'immagine, utilizzando `collapse(2)` con l'obiettivo di aumentare lo spazio di iterazioni disponibile al runtime OpenMP. Invece di distribuire soltanto `outputHeight` righe, OpenMP avrebbe potuto distribuire l'intero spazio `outputHeight × outputWidth` dei pixel.

In generale, però, si è osservato **un leggero peggioramento delle prestazioni**. Questo è motivabile con il fatto che il parallelismo applicato sul solo ciclo `y` fosse già ampiamente sufficiente data la dimensione delle immagini (> 4000x2000 pixel), e pertanto l'introduzione di `collapse` non risolve una carenza di lavoro da distribuire.

#### Riduzione SIMD sul kernel

Un'altra fetta di esperimenti ha riguardato i due cicli interni del kernel, i.e. lungo le righe (`j`) e le colonne (`i`) del kernel:

```cpp
for (unsigned int j = 0; j < order; ++j) {
    for (unsigned int i = 0; i < order; ++i) {
        ...
    }
}
```

Poiché questi cicli accumulano i valori di `channelRed`, `channelGreen` e `channelBlue`, si prestano particolarmente bene per l'applicazione del *pattern di riduzione*. Affinché potesse essere applicato senza dover generare ulteriori thread mediante un’altra direttiva `for` (oltre a quella già applicata al ciclo `y` — approccio che, sperimentalmente, ha mostrato una drastica riduzione delle prestazioni), è stato considerato l’utilizzo della direttiva `simd`, in modo da vettorizzare il ciclo interno senza ricorrere al worksharing.
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

**Questa forma ha ottenuto miglioramenti** veramente consistenti, **di oltre 2x per i kernel più grossi**. A tal proposito, si è voluto etichettare tale versione parallela come *Parallel#1*, cioè la migliore di questa prima parte di esperimenti, i cui risultati sono riportati in [Parallel#1's Experimental Results](https://github.com/marcopaglio/kip-parallel-OpenMP/tree/parallel%231/SoA#experimental-results "Experimental results section for the second parallelization of kip-parallel-openMP's SoA version"). Questa versione si distingue inoltre per un'altra piccola ottimizzazione: i calcoli invarianti di `posBase` e `kwBase` vengono eseguiti fuori dal ciclo `i` in modo da ridurre l'aritmetica degli indici.

#### Divisione manuale del lavoro tra thread

Il passo successivo è stato quello di provare ad eliminare completamente il worksharing construct `omp for`, mantenendo la sola regione `parallel`, e calcolando manualmente per ogni thread un intervallo di righe.

La prima versione è stata:

```cpp
lowerBound = outputHeight / nthreads * thread_id;

upperBound =
    thread_id == nthreads - 1
    ? outputHeight
    : outputHeight / nthreads * (thread_id + 1);
```

L'obiettivo era verificare se eliminando l'intervento dello scheduler OpenMP fosse possibile ottenere prestazioni migliori. Il risultato è stato però **identico o minimamente peggiore**.

È stato quindi individuato un problema in questa prima versione: tutto il resto della divisione viene assegnato all'ultimo thread; ciò potrebbe rendere la suddivisione manuale vana dal momento che tutti i thread dovrebbero attendere la fine del maggior carico di lavoro dell'ultimo thread. Si è dunque proposto una modalità *fair*, in cui le righe residue vengono distribuite una per volta ai primi thread:

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

Questa modifica **ha prodotto piccoli miglioramenti in tutti gli esperimenti rispetto alla prima divisione manuale**, confermando che il precedente squilibrio fosse reale. Tuttavia, anche dopo aver reso la partizione quasi perfettamente uniforme, **nel complesso la suddivisione manuale non ha portato miglioramenti**. Questo risultato dimostra che il vantaggio del dynamic scheduling non deriva semplicemente da un'implementazione inefficiente dello static scheduling di OpenMP.

### Restructuring

Dopo aver esplorato le possibilità fornite dalle *sole* direttive OpenMP, si è pensato di indagare sulla rielabolazione dell'algoritmo col fine di ottenere strutture che meglio si prestano alla parallelizzazione.

#### Blocking/Tiling 2D
Innazitutto, è stata introdotta una strategia di **blocking/tiling bidimensionale**. L'idea è di suddividere l'immagine in blocchi rettangolari di dimensione controllata utilizzando due iperparametri (`TILE_Y` e `TILE_X`);  la struttura generale è quindi del tipo:

```cpp
for (tileY)
    for (tileX)
        for (y nel tile)
            for (x nel tile)
                convoluzione
```

con parallelizzazione dei tile esterni tramite OpenMP e mantenendo la riduzione SIMD sul ciclo `i`.

Lo scopo principale del tiling sarebbe stata quella di migliorare la **locality dei dati**, favorendo il riutilizzo delle porzioni di immagine che appartengono a finestre di convoluzione vicine e riducendo il working set attivo per ciascun thread. Inoltre, rispetto a un semplice `collapse(2)` su `(y,x)`, il tile permetteva di mantenere una granularità più grossa e più controllabile.

Sono state provate diverse configurazioni, fra cui:

```text
TILE_Y = 4,  TILE_X = 64
TILE_Y = 8,  TILE_X = 128
TILE_Y = 16, TILE_X = 256
```

sia nella variante con `#pragma omp simd reduction(...)` sia nell'analoga variante ma senza la clausola `reduction`. 

I risultati, però, non hanno mostrato un miglioramento strutturale. Nella maggior parte dei casi le differenze rispetto alla versione senza tiling sono rimaste contenute in pochi punti percentuali, con alcuni piccoli miglioramenti e alcuni piccoli peggioramenti a seconda della combinazione immagine/kernel/tile: non è quindi emersa una configurazione universalmente migliore.

La conclusione è quindi che **il blocking 2D tradizionale non rappresenta il collo di bottiglia principale del programma**: la convoluzione continua infatti a mostrare una crescita significativa del tempo all'aumentare di `order`, e il tiling non modifica sostanzialmente questa tendenza. Questo esperimento è stato comunque utile perché ha suggerito che fosse necessario intervenire non soltanto sulla posizione dei dati in cache, ma sulla **struttura stessa dei loop e delle dipendenze interne al kernel**.

#### Tiling orizzontale

Un altro cambiamento ha voluto modificare radicalmente l'ordine dei loop. La struttura dal quale si è partiti è essenzialmente la seguente:

```cpp
for (y)
    for (x)
        for (j)
            for (i)
                accumula channelRed/Green/Blue
```

In questa forma, per ogni pixel venivano inizializzati tre accumulatori scalari e successivamente aggiornati `order²` volte. La nuova strategia ha invece introdotto un tile esclusivamente orizzontale:

```cpp
for (y)
    for (xTile)
        for (j)
            for (i)
                for (t)
```

dove `t` identifica i pixel all'interno del tile. Per ogni tile vengono creati tre vettori temporanei (`channelReds`, `channelGreens`, `channelBlues`) di dimensione `TILE_X`, e un singolo coefficiente del kernel che viene applicato in sequenza a tutti i pixel del tile. Questa versione presenta due vantaggi principali:

1) Fornisce al compilatore un ciclo interno estremamente semplice e regolare, con accessi contigui sia all'input sia agli accumulatori.
2) Elimina la lunga catena di dipendenza associata a un singolo accumulatore scalare, rendendo disponibile molto più instruction-level parallelism.

Il risultato è stato nettamente superiore a quello ottenuto con tutte le precedenti modifiche OpenMP. Già con `TILE_X = 64` e senza alcuna direttiva SIMD sul ciclo computazionale, il guadagno rispetto a *Parallel#1* è stato dell'ordine di circa $1.5–2.3\times$, a seconda della dimensione del kernel e dell'immagine. In particolare, il guadagno tende a essere più elevato coi kernel più piccoli e a ridursi con quelli più grandi, seppure rimanendo molto significativo.

Sulla macchina utilizzata, il miglior valore individuato sperimentalmente per l'iperparametro `TILE_X` è stato di `1024`, sebbene il vantaggio rispetto a `64` sia molto piccolo.

#### Applicazione delle direttive SIMD sulla versione `TILE_X`

Una volta ottenuta questa nuova struttura, sono state provate diverse configurazioni SIMD. 

È stato anzitutto inserito `#pragma omp simd` sul ciclo computazionale interno `t`. Contrariamente alle aspettative, la direttiva **non ha prodotto alcun miglioramento apprezzabile**: in alcuni test le prestazioni sono rimaste praticamente identiche, mentre in altri sono peggiorate leggermente. Questo suggerisce che il compilatore sia già in grado di ottimizzare e probabilmente auto-vettorizzare quel ciclo grazie alla nuova struttura molto più regolare.

È stata invece osservata una piccola ma abbastanza consistente utilità nell'applicare la stessa direttiva al ciclo finale che converte gli accumulatori floating point e salva i pixel:

```cpp
#pragma omp simd
    for (t) {
        reds[outTile] = getChannelAsUint8(channelReds[t]);
        greens[outTile] = getChannelAsUint8(channelGreens[t]);
        blues[outTile] = getChannelAsUint8(channelBlues[t]);
    }
```

Tale versione ottenuta è risultata la migliore dal punto di vista delle prestazione ed è stata pertanto marcata come *MainParallel*, e i risultati sperimentali sono stati aggiunti in [Experimental Results](#experimental-results).

#### Parallelizzazione dei tile

Dopo aver introdotto il tiling orizzontale `TILE_X`, è stato effettuato un ulteriore esperimento; invece di parallelizzare solamente le righe `y`, si è provato a distribuire direttamente le coppie $$(y,\ tile_x)$$ fra i thread OpenMP mediante `collapse(2)`, con l'obiettivo di aumentare il numero di unità di lavoro disponibili allo scheduler.

Questa strategia era potenzialmente interessante perché evita il problema del precedente `collapse(2)` applicato direttamente a `(y,x)`, dove l'unità di lavoro diventava un singolo pixel e la granularità risultava troppo fine. In questo caso, invece, ciascuna unità schedulata contiene circa `TILE_X` pixel e quindi una quantità di calcolo molto più elevata.

Sperimentalmente, tuttavia, **la parallelizzazione esplicita dei tile non ha prodotto benefici apprezzabili**, e le prestazioni sono rimaste sostanzialmente comparabili alla versione precedente. Questo risultato suggerisce che, per le immagini e il numero di thread utilizzati, la parallelizzazione del solo ciclo `y` mette già a disposizione una quantità sufficiente di lavoro.

### Experimental Results

The following tables summarizes the temporal measurements of convolutions on different images with different kernels, measured in release mode.

<table>
  <thead>
    <tr>
      <th colspan="3" rowspan="3">Execution Time<br>(Release mode)</th>
      <th colspan="12">Image Dimension</th>
    </tr>
    <tr>
      <th colspan="3">4K</th>
      <th colspan="3">5K</th>
      <th colspan="3">6K</th>
      <th colspan="3">7K</th>
    </tr>
    <tr>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="9"><strong>Kernel Dimension</strong></td>
      <td rowspan="4"><strong>Box Blurring</strong></td>
      <td><strong>7</strong></td>
      <td>0.04842</td>
      <td>0.03416</td>
      <td>0.03320</td>
      <td>0.06200</td>
      <td>0.05832</td>
      <td>0.05928</td>
      <td>0.09549</td>
      <td>0.09818</td>
      <td>0.09659</td>
      <td>0.13456</td>
      <td>0.13350</td>
      <td>0.13551</td>
    </tr>
    <tr>
      <td><strong>13</strong></td>
      <td>0.06750</td>
      <td>0.06757</td>
      <td>0.06898</td>
      <td>0.12669</td>
      <td>0.12724</td>
      <td>0.12720</td>
      <td>0.20207</td>
      <td>0.20178</td>
      <td>0.20306</td>
      <td>0.29487</td>
      <td>0.29504</td>
      <td>0.29867</td>
    </tr>
    <tr>
      <td><strong>19</strong></td>
      <td>0.12546</td>
      <td>0.12565</td>
      <td>0.12595</td>
      <td>0.23875</td>
      <td>0.23455</td>
      <td>0.23472</td>
      <td>0.37526</td>
      <td>0.37985</td>
      <td>0.38246</td>
      <td>0.54533</td>
      <td>0.55172</td>
      <td>0.54782</td>
    </tr>
    <tr>
      <td><strong>25</strong></td>
      <td>0.20849</td>
      <td>0.20443</td>
      <td>0.20543</td>
      <td>0.38263</td>
      <td>0.38251</td>
      <td>0.38294</td>
      <td>0.62168</td>
      <td>0.62205</td>
      <td>0.62164</td>
      <td>0.94424</td>
      <td>0.92882</td>
      <td>0.94158</td>
    </tr>
  </tbody>
</table>

It can be seen that the times recorded for inputs of the same size are very similar, indicating that the execution times of the operations are independent of the pixel values.

### Profiling Results

TODO
