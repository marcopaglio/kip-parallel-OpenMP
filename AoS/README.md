## Kip-parallel-OpenMP - AoS - Parallel#0 branch

This branch differs from the [AoS version of the main branch](https://github.com/marcopaglio/kip-parallel-OpenMP/AoS "AoS folder of kip-parallel-OpenMP's main branch") for the implementation of `ImageProcessing::convolution` method and the way it is parallelised.<br>

In particular, this parallel version, named *Parallel#0*, is the first directly derived from [kip-sequential’s AoS](https://github.com/marcopaglio/kip-sequential/AoS "AoS folder of kip-sequential’s main branch"), and is also the simplest. In fact, the parallelisation process here is limited to the following changes:

- Parallel region and worksharing over the outer loop `y`, i.e. along the rows of the image.
- Selection of the `dynamic` scheduler.
- Choice between `shared` and `firstprivate` for the variables used by the threads.

The overall structure is essentially as follows:

```cpp
#pragma omp parallel for schedule(dynamic) default(none) \
shared(pixels, originalData, outputHeight) \
firstprivate(outputWidth, order, kernelWeights)
for (y) {
    for (x) {
        for (j) {
            for (i) {
                // accumulate channel Red/Green/Blue
            }
        }
    }
}
```

In this form, three scalar accumulators are initialised for each pixel and subsequently updated `order²` times. The strategy differs from the [AoS version of the main branch](https://github.com/marcopaglio/kip-parallel-OpenMP/AoS "AoS folder of kip-parallel-OpenMP's main branch") for the introduction of horizontal tiling, i.e:

```cpp
#pragma omp parallel for schedule(dynamic) default(none) \
shared(pixels, originalData, outputHeight, TILE_X) \
firstprivate(outputWidth, order, kernelWeights)
for (y)
    for (xTile)
        for (j)
            for (i)
                for (t)
```

where `t` identifies the pixels within the tile. For each tile, three temporary vectors (`channelReds`, `channelGreens`, `channelBlues`) of size `TILE_X` are created, along with a single kernel coefficient which is applied sequentially to all the pixels in the tile.

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
      <td>0.22571</td>
      <td>0.20754</td>
      <td>0.20577</td>
      <td>0.38316</td>
      <td>0.38474</td>
      <td>0.38486</td>
      <td>0.63534</td>
      <td>0.66016</td>
      <td>0.65438</td>
      <td>0.951797</td>
      <td>0.93239</td>
      <td>0.96352</td>
    </tr>
    <tr>
      <td><strong>13</strong></td>
      <td>0.55291</td>
      <td>0.56398</td>
      <td>0.569123</td>
      <td>1.06067</td>
      <td>1.06201</td>
      <td>1.07798</td>
      <td>1.73473</td>
      <td>1.72339</td>
      <td>1.88896</td>
      <td>2.53782</td>
      <td>2.5614</td>
      <td>2.53318</td>
    </tr>
    <tr>
      <td><strong>19</strong></td>
      <td>1.14786</td>
      <td>1.11774</td>
      <td>1.12881</td>
      <td>2.12961</td>
      <td>2.10217</td>
      <td>2.10483</td>
      <td>3.45105</td>
      <td>3.43448</td>
      <td>3.49643</td>
      <td>5.02615</td>
      <td>5.02368</td>
      <td>5.00482</td>
    </tr>
    <tr>
      <td><strong>25</strong></td>
      <td>1.88709</td>
      <td>1.91398</td>
      <td>1.88327</td>
      <td>3.58709</td>
      <td>3.52798</td>
      <td>3.52466</td>
      <td>5.70262</td>
      <td>5.71054</td>
      <td>5.85042</td>
      <td>8.41449</td>
      <td>8.39587</td>
      <td>8.54511</td>
    </tr>
  </tbody>
</table>

It can be seen that the times recorded for inputs of the same size are very similar, indicating that the execution times of the operations are independent of the pixel values.
