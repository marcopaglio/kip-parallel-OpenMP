## Kip-parallel-OpenMP - AoS

This is the *Array-of-Structures* version of **kip-parallel-OpenMP**. Alternative versions can be found at:

- [SoA folder](../SoA "SoA version of kip-parallel-OpenMP").
- [Parallel#0 branch](https://github.com/marcopaglio/kip-sequential/tree/parallel%230/AoS "First parallelization of kip-parallel-openMP's AoS version")
- [Parallel#1 branch](https://github.com/marcopaglio/kip-sequential/tree/parallel%231/AoS "Second parallelization of kip-parallel-openMP's AoS version")

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
shared(pixels, originalData, outputHeight) \
firstprivate(outputWidth, order, kernelWeights)
```

The general approach taken was to distinguish between large variables or those naturally shared among all threads, and scalar variables which are small and used very frequently; for the latter, `firstprivate` may be a reasonable choice because each thread receives its own copy initialized with the original value, whereas for larger data structures, `shared` avoids the cost of a private copy. In this regard:

- For the output vectors (`pixels`), the `shared` option is both natural and necessary. All threads must contribute to the same final result, but each `(y,x)` iteration writes to a different location, so there can be no race conditions.
- The same reasoning holds for `originalData`, though for different reasons. This vector is read-only and is much larger than the other variables in the parallel region. `shared` is therefore the most sensible choice: there is no risk of a data race due to concurrent reads, and a separate copy per thread would potentially incur a very high cost in terms of memory and initialization.
- `outputHeight` has been kept as `shared`. It is a read-only scalar value and is primarily used as a bound for the outer iteration space. From a correctness standpoint, it could also be `firstprivate`, but since it is rarely used within the actual kernel, there is no concrete reason to create a private copy for each thread.
- For `outputWidth` and `order`, instead, `firstprivate` was chosen. The reason is that these are very small scalars, so the cost of duplication is negligible; furthermore, they are frequently used in the parallel region. 
- The most ambiguous case was `kernelWeights`: in fact, it is not a scalar but a container of $order^2$ coefficients. Since it is read-only, one might think that `shared` is preferable for `kernelWeights`, in order to avoid the cost of constructing and copying the vector for each thread. However, the experimental results show small variations depending on the kernel size: the most accurate conclusion is that there is no clear winner, but that this should continue to be taken into account for future experiments. In fact, these differences seem to be attributable more to microarchitectural effects than to a fundamental difference in the OpenMP model. In conclusion, for the time being, we have opted to keep it private for each thread, following the principle of privatizing as much as possible.

Overall, these first two modifications have yielded excellent results compared to the sequential version of the problem; therefore, the version of the program described so far is identified as *Parallel#0*. The corresponding performance results are reported in [Parallel#0's Experimental Results](https://github.com/marcopaglio/kip-parallel-OpenMP/tree/parallel%230/AoS#experimental-results "Experimental results section for the first parallelization of kip-parallel-openMP's AoS version").

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

2) By applying SIMD directly to `i`, i.e. to consecutive elements in the same row of both image and kernel:

```cpp
for (unsigned int j = 0; j < order; ++j) {
    const unsigned int posBase = y + j;
    const unsigned int kwBase = j * order;
#pragma omp simd reduction(+:channelRed, channelGreen, channelBlue)
    for (unsigned int i = 0; i < order; ++i) {
        ...
    }
}
```

In both cases, **performance remained the same**, mainly because in the AoS form, the compiler is unable to vectorize the loops. As evidence of this, it returns the warning `loop not vectorized` during compilation, meaning that the directives themselves are not taken into account.

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

The goal was to determine whether eliminating the OpenMP scheduler’s intervention would result in better performance. However, the results showed a **clear deterioration**.

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

#### Horizontal Tiling

Another change was intended to radically alter the order of the loops. The initial structure was essentially as follows:

```cpp
for (y)
    for (x)
        for (j)
            for (i)
                accumulate channelRed/Green/Blue
```

In this approach, three scalar accumulators were initialized for each pixel and subsequently updated $order^2$ times. The new strategy, on the other hand, introduced an exclusively horizontal tile:

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

Results show a general 10% improvement compared to that achieved with all previous OpenMP modifications (i.e. *Parallel#0*). In particular, the performance gain tends to be higher with smaller kernels and to decrease with larger ones, although it remains very significant.

On the machine used, the best value identified experimentally for the hyperparameter `TILE_X` was `1024`. This version proved to be the best in terms of performance and was therefore labeled *MainParallel*, and the experimental results were added to [Experimental Results](#experimental-results).

#### SIMD directive on `TILE_X`

Once this new structure was obtained, several SIMD configurations were tested; however, the compiler once again generated the warning `loop not vectorized`, so every attempt to improve performance without additional worksharing proved futile.

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
      <td>0.20504</td>
      <td>0.16468</td>
      <td>0.16378</td>
      <td>0.30609</td>
      <td>0.30760</td>
      <td>0.30952</td>
      <td>0.53501</td>
      <td>0.51284</td>
      <td>0.50828</td>
      <td>0.74517</td>
      <td>0.75032</td>
      <td>0.74818</td>
    </tr>
    <tr>
      <td><strong>13</strong></td>
      <td>0.49079</td>
      <td>0.48147</td>
      <td>0.47891</td>
      <td>0.90793</td>
      <td>0.90491</td>
      <td>0.90464</td>
      <td>1.4898</td>
      <td>1.48822</td>
      <td>1.48679</td>
      <td>2.19345</td>
      <td>2.18766</td>
      <td>2.17106</td>
    </tr>
    <tr>
      <td><strong>19</strong></td>
      <td>1.00431</td>
      <td>1.00307</td>
      <td>1.00511</td>
      <td>1.91239</td>
      <td>1.91459</td>
      <td>1.93517</td>
      <td>3.08409</td>
      <td>3.08064</td>
      <td>3.08502</td>
      <td>4.54001</td>
      <td>4.52739</td>
      <td>4.52115</td>
    </tr>
    <tr>
      <td><strong>25</strong></td>
      <td>1.73494</td>
      <td>1.73553</td>
      <td>1.75971</td>
      <td>3.34289</td>
      <td>3.29614</td>
      <td>3.29618</td>
      <td>5.28777</td>
      <td>5.28481</td>
      <td>5.28199</td>
      <td>7.75955</td>
      <td>7.80973</td>
      <td>7.84652</td>
    </tr>
  </tbody>
</table>

It can be seen that the times recorded for inputs of the same size are very similar, indicating that the execution times of the operations are independent of the pixel values.
