## Kip-parallel-OpenMP - SoA - Parallel#1 branch

This branch differs from the [SoA version of the main branch](https://github.com/marcopaglio/kip-parallel-OpenMP/SoA "SoA folder of kip-parallel-OpenMP's main branch") for the implementation of `ImageProcessing::convolution` method and the way it is parallelised.<br>

In particular, this parallel version, named *Parallel#1*, increases the parallelisation of [Parallel#0 branch](https://github.com/marcopaglio/kip-sequential/tree/parallel%231/SoA "SoA folder of kip-parallel-openMP’s Parallel#0 branch") with the application of the SIMD reduction to the inner cycle `i`, i.e. to consecutive elements in the same row of the image and the kernel.

The overall structure is essentially as follows:

```cpp
#pragma omp parallel for schedule(dynamic) default(none) \
shared(pixels, originalData, outputHeight) \
firstprivate(outputWidth, order, kernelWeights)
for (y)
    for (x)
        for (j)
#pragma omp simd reduction(+:channelRed, channelGreen, channelBlue)
            for (i)
                // accumulate channel Red/Green/Blue
```

This strategy differs from the [SoA version of the main branch](https://github.com/marcopaglio/kip-parallel-OpenMP/SoA "SoA folder of kip-parallel-OpenMP's main branch") for the introduction of horizontal tiling, i.e:

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
      <td>0.07326</td>
      <td>0.06666</td>
      <td>0.06666</td>
      <td>0.12211</td>
      <td>0.12447</td>
      <td>0.12129</td>
      <td>0.21209</td>
      <td>0.20492</td>
      <td>0.20264</td>
      <td>0.30077</td>
      <td>0.29933</td>
      <td>0.30445</td>
    </tr>
    <tr>
      <td><strong>13</strong></td>
      <td>0.14543</td>
      <td>0.14777</td>
      <td>0.15370</td>
      <td>0.28069</td>
      <td>0.27642</td>
      <td>0.27682</td>
      <td>0.45134</td>
      <td>0.44818</td>
      <td>0.44401</td>
      <td>0.65513</td>
      <td>0.65689</td>
      <td>0.65256</td>
    </tr>
    <tr>
      <td><strong>19</strong></td>
      <td>0.22981</td>
      <td>0.22341</td>
      <td>0.22047</td>
      <td>0.42488</td>
      <td>0.42635</td>
      <td>0.42701</td>
      <td>0.66110</td>
      <td>0.66089</td>
      <td>0.67528</td>
      <td>1.02425</td>
      <td>1.03523</td>
      <td>1.02628</td>
    </tr>
    <tr>
      <td><strong>25</strong></td>
      <td>0.29645</td>
      <td>0.30961</td>
      <td>0.30294</td>
      <td>0.54840</td>
      <td>0.54674</td>
      <td>0.55896</td>
      <td>0.91207</td>
      <td>0.90023</td>
      <td>0.91404</td>
      <td>1.42318</td>
      <td>1.42543</td>
      <td>1.45833</td>
    </tr>
  </tbody>
</table>

It can be seen that the times recorded for inputs of the same size are very similar, indicating that the execution times of the operations are independent of the pixel values.
