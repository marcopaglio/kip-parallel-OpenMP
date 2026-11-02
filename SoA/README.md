## Kip-parallel-OpenMP - SoA

This is the *Structure-of-Arrays* version of **kip-parallel-OpenMP**. Alternative versions can be found at:

- [AoS folder](../AoS "AoS version of kip-parallel-OpenMP")
- [Parallel#0 branch](https://github.com/marcopaglio/kip-sequential/tree/parallel%230/SoA "First parallelization of kip-parallel-openMP's SoA version")
- [Parallel#1 branch](https://github.com/marcopaglio/kip-sequential/tree/parallel%231/SoA "Second parallelization of kip-parallel-openMP's SoA version")

### Parallelization iter

The parallelization process involved only the `ImageProcessing::convolution` method and followed a path that was fairly consistent with the preliminary results obtained along the way, which can be summarized as follows: first, OpenMP parallelization was applied to the loop over the rows; next, scheduling and variable privatization were explored; then `collapse` and the differences with kernel vectorization were examined; and finally, the work was manually partitioned.

#### Parallelization of the External Cycle and Schedulers Comparison

The first natural parallel version is the one in which OpenMP distributes the iterations of the outer loop over `y`, i.e. across the rows of the image:

```cpp
#pragma omp parallel for schedule(...)
for (unsigned int y = 0; y < outputHeight; ++y) {
    for (unsigned int x = 0; x < outputWidth; ++x) {
        ...
    }
}
```

The initial assumption was that `schedule(static)` was the best choice. In fact, from an algorithmic standpoint, each iteration of `y` essentially performs the same work: it traverses `outputWidth` pixels and, for each pixel, performs a convolution of `order × order` elements. `static` should therefore have provided a good balance with less overhead than `dynamic` and `guided`. However, the measurements showed something different:
- Already in the initial experiments, it was observed that **`dynamic` and `guided` were faster than `static`**, with a performance advantage of about 10%; this result suggests that *a uniform number of operations does not necessarily result in perfectly uniform execution times across threads*.
- Further evidence supporting this impression was obtained by comparing **`static,1` and `static`**, which **delivered nearly identical performance**. The fact that `static,1` does not perform significantly better than the standard `static` is interesting because the two distribute the lines very differently: the former cyclically, the latter typically in contiguous blocks. This makes the hypothesis that the advantage of `dynamic` depends primarily on the specific position of the lines assigned to the threads less convincing. The result points more toward `dynamic`’s ability to absorb small differences in the actual execution times of the threads.

Finally, we also attempted to manually control the chunk size, using values derived from `outputHeight / (omp_get_num_threads() * min_chunk)` with `min_chunk` set to 1, 2, 4, and 8. In this case, **the explicitly set chunk sizes yielded worse results than the size chosen by default by the respective schedulers**. Consequently, the configuration that has established itself as the best remains:

```cpp
#pragma omp parallel for schedule(dynamic)
```

#### Choosing between `shared` and `firstprivate`

Another part of the experiment involved the classification of variables in the OpenMP region, which is made explicit using `default(none)`. The current configuration is:

```cpp
shared(reds, greens, blues, originalReds, originalGreens, originalBlues, outputHeight) \
firstprivate(width, outputWidth, order, kernelWeights)
```

The general approach taken was to distinguish between large variables or those naturally shared among all threads, and scalar variables which are small and used very frequently; for the latter, `firstprivate` may be a reasonable choice because each thread receives its own copy initialized with the original value, whereas for larger data structures, `shared` avoids the cost of a private copy. In this regard:

- For the output vectors (`reds`, `greens`, `blues`), the `shared` option is both natural and necessary. All threads must contribute to the same final result, but each `(y,x)` iteration writes to a different location, so there can be no race conditions.
- The same reasoning holds for `originalReds`, `originalGreens`, and `originalBlues`, though for different reasons. These vectors are read-only and are much larger than the other variables in the parallel region. `shared` is therefore the most sensible choice: there is no risk of a data race due to concurrent reads, and a separate copy per thread would potentially incur a very high cost in terms of memory and initialization.
- `outputHeight` has been kept as `shared`. It is a read-only scalar value and is primarily used as a bound for the outer iteration space. From a correctness standpoint, it could also be `firstprivate`, but since it is rarely used within the actual kernel, there is no concrete reason to create a private copy for each thread.
- For `width`, `outputWidth`, and `order`, instead `firstprivate` was chosen. The reason is that these are very small scalars, so the cost of duplication is negligible; furthermore, they are frequently used in the parallel region.
- The most ambiguous case was `kernelWeights`: in fact, it is not a scalar but a container of $order^2$ coefficients. Since it is read-only, one might think that `shared` is preferable for `kernelWeights`, in order to avoid the cost of constructing and copying the vector for each thread. However, the experimental results show small variations depending on the kernel size: the most accurate conclusion is that there is no clear winner, but that this should continue to be taken into account for future experiments. In fact, these differences seem to be attributable more to microarchitectural effects than to a fundamental difference in the OpenMP model. In conclusion, for the time being, we have opted to keep it private for each thread, following the principle of privatizing as much as possible.

Overall, these first two modifications have yielded excellent results compared to the sequential version of the problem; therefore, the version of the program described so far is identified as *Parallel#0*. The corresponding performance results are reported in [Parallel#0's Experimental Results](https://github.com/marcopaglio/kip-parallel-OpenMP/tree/parallel%230/SoA#experimental-results "Experimental results section for the first parallelization of kip-parallel-openMP's SoA version").

#### Using `collapse`

Next, we attempted to take advantage of the complete independence of the first two loops, i.e. the movement along the rows (`y`) and columns (`x`) of the image, by using `collapse(2)`, with the goal of increasing the number of iterations available to the OpenMP runtime. Instead of distributing only `outputHeight` rows, OpenMP could have distributed the entire `outputHeight × outputWidth` pixel space.

In general, however, **a slight performance degradation** was observed. This can be explained by the fact that the parallelism applied to the `y` loop alone was already more than sufficient given the size of the images (> 4000x2000 pixels), and therefore the introduction of `collapse` does not address a shortage of work to be distributed.

#### SIMD Reduction on the Kernel

Another set of experiments focused on the kernel's two inner loops, i.e. along the rows (`j`) and columns (`i`) of the kernel:

```cpp
for (unsigned int j = 0; j < order; ++j) {
    for (unsigned int i = 0; i < order; ++i) {
        ...
    }
}
```

Since these loops accumulate the values of `channelRed`, `channelGreen`, and `channelBlue`, they are particularly well-suited for applying the *reduction pattern*. To apply this pattern without having to generate additional threads by using another `for` directive (in addition to the one already applied to the `y` loop — an approach that, in testing, resulted in a drastic drop in performance), we considered using the `simd` directive to vectorize the inner loop without resorting to worksharing.
We tested two main approaches:

1. Through the combined use of `collapse`:

```cpp
#pragma omp simd collapse(2) reduction(+:channelRed, channelGreen, channelBlue)
for (unsigned int j = 0; j < order; ++j) {
    for (unsigned int i = 0; i < order; ++i) {
        ...
    }
}
```

However, **performance has deteriorated significantly, and this deterioration has increased as the size of the inputs has grown**.

2) By applying SIMD directly to `i`, i.e. to consecutive elements in the same row of both image and kernel:

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

**This version has shown truly significant improvements, more than 2x performance for the largest kernels**. In this regard, we have chosen to label this parallel version as *Parallel#1*, i.e. the best of this first set of experiments, the results of which are reported in [Parallel#1's Experimental Results](https://github.com/marcopaglio/kip-parallel-OpenMP/tree/parallel%231/SoA#experimental-results "Experimental results section for the second parallelization of kip-parallel-openMP's SoA version"). This version also features another small optimization: the invariant calculations of `posBase` and `kwBase` are performed outside the `i` loop in order to reduce index arithmetic.

#### Manual worksharing among threads

The next step was to try to completely eliminate the `omp for` worksharing construct, retaining only the `parallel` region, and manually calculating a range of rows for each thread.
The first version was:

```cpp
lowerBound = outputHeight / nthreads * thread_id;

upperBound =
    thread_id == nthreads - 1
    ? outputHeight
    : outputHeight / nthreads * (thread_id + 1);
```

The goal was to determine whether eliminating the OpenMP scheduler’s intervention would result in better performance. However, the result was **identical or only slightly worse**.

A problem was therefore identified in this first version: the rest of the division is assigned to the last thread; this could render manual partitioning ineffective, since all threads would have to wait for the last thread to finish its heavier workload. A *fair* mode was therefore proposed, in which the remaining lines are distributed one at a time to the first threads:

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

This change **resulted in slight improvements in all experiments compared to the initial manual partitioning**, confirming that the previous imbalance was real. However, even after making the partitioning nearly perfectly uniform, **overall, the manual partitioning did not lead to any improvements**. This result demonstrates that the advantage of dynamic scheduling does not simply stem from an inefficient implementation of OpenMP’s static scheduling.

### Restructuring

After exploring the possibilities offered by *only* the OpenMP directives, we decided to investigate redesigning the algorithm in order to obtain data structures that are better suited for parallelization.

#### Blocking/Tiling 2D

First, a **two-dimensional blocking/tiling** strategy was introduced. The idea is to divide the image into rectangular blocks of a controlled size using two hyperparameters (`TILE_Y` and `TILE_X`), while maintaining parallelization of the outer blocks via OpenMP and SIMD reduction on loop `i`. The general structure is therefore as follows:

```cpp
for (tileY)
    for (tileX)
        for (y in tile)
            for (x in tile)
                convolution
```

The main purpose of tiling was to improve **data locality**, promoting the reuse of image portions belonging to neighboring convolution windows and reducing the active working set for each thread. Furthermore, compared to a simple `collapse(2)` on `(y,x)`, the tile allowed for a coarser and more controllable granularity.

Several configurations were tested, both in the variant with `#pragma omp simd reduction(...)` and in the analogous variant without the `reduction` clause, including:

```text
TILE_Y = 4,  TILE_X = 64
TILE_Y = 8,  TILE_X = 128
TILE_Y = 16, TILE_X = 256
```

The results, however, did not show any structural improvement. In most cases, the differences compared to the version without tiling were limited to a few percentage points, with some minor improvements and some minor deteriorations depending on the image/kernel/tile combination: therefore, no universally superior configuration emerged.

The conclusion, therefore, is that **traditional 2D blocking is not the program’s main bottleneck**: in fact, convolution continues to show a significant increase in execution time as `order` increases, and tiling does not substantially alter this trend. This experiment was nevertheless useful because it suggested that it was necessary to optimize not only the positioning of data in the cache, but also the **structure of the loops and internal dependencies within the kernel**.

#### Horizontal Tiling

Another change was intended to radically alter the order of the loops. The initial structure was essentially as follows:

```cpp
for (y)
    for (x)
        for (j)
            for (i)
                accumulate channelRed/Green/Blue
```

In this approach, three scalar accumulators were initialized for each pixel and subsequently updated `order²` times. The new strategy, on the other hand, introduced an exclusively horizontal tile:

```cpp
for (y)
    for (xTile)
        for (j)
            for (i)
                for (t)
                    accumulate channelRed/Green/Blue
```

where `t` identifies the pixels within the tile. For each tile, three temporary vectors (`channelReds`, `channelGreens`, `channelBlues`) of size `TILE_X` are created, along with a single kernel coefficient that is applied sequentially to all pixels in the tile. The visual difference between the image before and after the application of horizontal tiling can be seen in [Fig. 1](#figure-1).

<p align="center">
  <img id="figure-1" src="/../assets/horizontal_tiling_before_and_after.png" alt="Visualization of before and after applying horizontal tiling." title="Horizontal tiling visualization" width="90%" />
</p>

This version offers three main advantages:

1) It allows the *first touch* strategy to be applied to threads on channel accumulators, promoting optimal memory access and the use of local memory (which in turn can lead to superlinear speedups compared to the sequential version).
2) The row-based partitioning provides the compiler with an extremely simple and regular internal loop, with contiguous accesses to both the input and the accumulators, significantly reducing the potential problem of *false sharing* (although this solution does not guarantee cache-aligned memory banks, even if `TILE_X` is chosen as a multiple of that size).
3) It eliminates the long dependency chain associated with a single scalar accumulator, making much more instruction-level parallelism available.

The result was significantly better than that obtained with all previous OpenMP modifications. Even with `TILE_X = 64` and without any SIMD directives in the computational loop, the performance gain compared to *Parallel#1* was on the order of approximately $1.5–2.3\times$, depending on the size of the kernel and the image. In particular, the performance gain tends to be higher with smaller kernels and to decrease with larger ones, although it remains very significant.

On the machine used, the best value identified experimentally for the hyperparameter `TILE_X` was `1024`, although the advantage over `64` is very small.

#### SIMD directive on `TILE_X`

Once this new structure was obtained, several SIMD configurations were tested.

First, `#pragma omp simd` was inserted into the internal computational loop `t`. Contrary to expectations, the directive **did not produce any noticeable improvement**: in some tests, performance remained virtually identical, while in others it deteriorated slightly. This suggests that the compiler is already able to optimize and likely auto-vectorize that loop thanks to the new, much more regular structure.

On the other hand, a small but fairly consistent benefit was observed when applying the same directive to the final loop that converts the floating-point accumulators and saves the pixels:

```cpp
#pragma omp simd
    for (t) {
        reds[outTile] = getChannelAsUint8(channelReds[t]);
        greens[outTile] = getChannelAsUint8(channelGreens[t]);
        blues[outTile] = getChannelAsUint8(channelBlues[t]);
    }
```

This version proved to be the best in terms of performance and was therefore labeled *MainParallel*, and the experimental results were added to [Experimental Results](#experimental-results).

#### Tile Parallelization

After introducing horizontal tiling `TILE_X`, a further experiment was conducted; instead of parallelizing only the `y` rows, an attempt was made to distribute the pairs $$(y, tile_x)$$ directly among the OpenMP threads using `collapse(2)`, with the goal of increasing the number of work units available to the scheduler.

This strategy was potentially interesting because it avoids the problem of the previous `collapse(2)` applied directly to `(y,x)`, where the work unit became a single pixel and the granularity was too fine. In this case, however, each scheduled unit contains approximately `TILE_X` pixels and thus a much higher computational load.

Experimentally, however, **explicit tile parallelization did not yield appreciable benefits**, and performance remained essentially comparable to the previous version. This result suggests that, for the images and the number of threads used, parallelizing only the `y` loop already provides a sufficient amount of work.


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

### Profiling

While profiling in the sequential version is mainly used to identify programme hotspots, in the parallel version its primary aim is to identify potential improvements. In this regard, the `profile.cpp` in the OpenMP version differs from the sequential version in 3 main ways:

- It utilises the maximum number of threads made available by the processors.
- It focuses profiling solely on hotspot functions rather than on the entire programme.
  + In this project, the by far most significant hotspot function is `ImageProcessing::convolution`.
  + VTune allows partial profiling to be applied via the `__itt_pause` and `__itt_resume` APIs in the `ittnotify.h` library; however, as these introduce significant overhead, Intel recommends calling them at a frequency of no less than 1 Hz.
- Execute the hotspot function multiple times (`numThreads` times) in succession so that sufficient data is generated for analysis (the parallel execution typically takes few tenths of a second).

#### Results

The *Performance Snapshot* for the OpenMP programme shows a positive overall behaviour:

- **Logical Core Utilization** exceeds **80%**, while **Physical Core Utilization** reaches **72%**.
- **Microarchitecture Usage** stands at **65%**, with excellent **Instruction Retiring rates** for both Performance-cores (**68%**) and Efficient-cores (**58%**).
- **Vectorization** of Packed Floating Point (FP) Operations **is 99.6%**.

A more in-depth analysis of the `ImageProcessing::convolution` function also reveals:

- For *Memory Access*, there is a high number of loads (~123 billions) and stores (~62 billions), accompanied by a **small number of LLC Miss Counts** (~275000) and **almost absent Memory Bound** (**3.1%**). For all these reasons, it can be concluded that the programme is not limited by the memory bandwidth.
- For *Microarchitecture Exploration* there is a clear distinction between P-cores and E-cores; in particular, the former show a **Back-End Bound** lower than the latter (**14.6%** and **26.4%**), with **Core Bound** being **the predominant** factor in both cases. This may indicate that the machine ran out of out-of-order (OOO) resources, that certain execution units are overloaded, or that dependencies in the programme’s data or instruction flow are limiting performance. Furthermore, the **Load Bound at 13.2%** suggests a problem linked to memory accesses, but the **Stores Bound at 0.4%** excludes significant write problems, such as *false sharing*. Other details can be found in the [Fig. 2](#figure-2).

<p align="center">
  <img id="figure-2" src="/../assets/vtune_omp_rel_ue_1ms_soa.png" alt="Screenshot of Microarchitecture Exploration profiling summary." title="Microarchitecture Exploration summary" width="40%"/>
</p>

- The *Threading* analysis hightlights the **low utilization of logical cores**, which is standing **at just 23%**, with an average utilization of **only 4 cores** (out of 16 available!). The **Preemption Wait Time of 6.370s** suggests that the main cause is an excessive use of threads for programme execution; indeed, a significant Preemption Wait Time can signal a thread oversubscription problem either because of oversubscription of application threads or a conflict with OS threads or other applications on the system. On the other hand, the **Inactive Sync Wait Time is almost zero**. This result can also be observed graphically in the strong and weak scaling calculations, where a marked drop in performance is evident when scaling from 4 to 8 threads, and even more so when scaling from 8 to 16 threads, as shown in the [Fig. 3](#figure-3). However, switching to a higher number of threads results in a performance benefit, so it is certainly worth keeping.

<p align="center">
  <img id="figure-3" src="/../assets/soa_parallel2_strong_weak_6_19.png" alt="Strong and weak scaling curves for a 6K image and a kernel of order 19." title="Strong and weak scaling curves"/>
</p>

#### Consequences

As revealed by the profiling, the main problem can be attributed to *load imbalance*, which results in idle threads. Consequently, the main steps to further improve parallelism in the programme could be to review the scheduling policies.

In this regard, greater control can be achieved by adjusting the *thread affinity*. The main idea might be to use only the P-cores, with the aim of balancing thread utilization and eliminating the Back-End Bound issues associated with the E-cores. Assuming the number of P-cores is known (e.g. `p_core_number=10`), on an Intel machine this behaviour can be set via the following configuration, which must be placed before any other operation that might have initialised the OpenMP runtime:

```cpp
#ifdef _WIN32
    _putenv_s("KMP_HW_SUBSET", "10c:intel_core,1t");
    _putenv_s("OMP_NUM_THREADS", "10");
    _putenv_s("OMP_PROC_BIND", "close");
#elif __unix__
    setenv("KMP_HW_SUBSET", "10c:intel_core,1t", 1);
    setenv("OMP_NUM_THREADS", "10", 1);
    setenv("OMP_PROC_BIND", "close", 1);
#endif
```

As might be expected from the graphs in the [Fig. 2](#figure-2), this change reduces performance, since a greater number of threads (slightly) improves the total execution time.

Secondly, the profiling suggests that the structure and logic of data access should be further improved to facilitate intuitive memory access by the programme, for example by ensuring contiguous accesses and avoiding irregular accesses that make it more difficult to utilize caching and vectorization.

The SoA architecture has already exploited most of the potential benefits arising from these aspects. 
A further improvement could be achieved by removing `std::vector` in `Image` class, but this change is not taken into account because it alters the main structure of the project, making it more difficult to compare with the sequential version. 
Another startegy could be using cache-aligned data structures, but this also requires using of simpler data structures, such as:

```cpp
auto* reds = static_cast<uint8_t*>(_aligned_malloc(size, CACHE_LINE_SIZE));
auto* greens = static_cast<uint8_t*>(_aligned_malloc(size, CACHE_LINE_SIZE));
auto* blues = static_cast<uint8_t*>(_aligned_malloc(size, CACHE_LINE_SIZE));
```

where `CACHE_LINE_SIZE=64`, depending on your cache configuration.
