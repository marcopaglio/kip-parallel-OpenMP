## Kip-parallel-OpenMP - SoA - Parallel#0 branch

This branch differs from the [SoA version of the main branch](https://github.com/marcopaglio/kip-parallel-OpenMP/SoA "SoA folder of kip-parallel-OpenMP's main branch") for the implementation of `ImageProcessing::convolution` method and the way it is parallelised.<br>

In particular, this parallel version, named *Parallel#0*, is the first directly derived from [kip-sequential’s SoA](https://github.com/marcopaglio/kip-sequential/SoA "SoA folder of kip-sequential’s main branch"), and is also the simplest. In fact, the parallelisation process here is limited to the following changes:

- Parallel region and worksharing over the outer loop `y`, i.e. along the rows of the image.
- Selection of the `dynamic` scheduler.
- Choice between `shared` and `firstprivate` for the variables used by the threads.

The overall structure is essentially as follows:

```cpp
#pragma omp parallel for schedule(dynamic) default(none) \
shared(pixels, originalData, outputHeight) \
firstprivate(outputWidth, order, kernelWeights)
for (y)
    for (x)
        for (j)
            for (i)
                // accumulate channel Red/Green/Blue
```

In this form, three scalar accumulators are initialised for each pixel and subsequently updated `order²` times. The strategy differs from the [SoA version of the main branch](https://github.com/marcopaglio/kip-parallel-OpenMP/SoA "SoA folder of kip-parallel-OpenMP's main branch") for the introduction of horizontal tiling, i.e:

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
      <td>0.070517</td>
      <td>0.06871</td>
      <td>0.06756</td>
      <td>0.12581</td>
      <td>0.12900</td>
      <td>0.12359</td>
      <td>0.19670</td>
      <td>0.19746</td>
      <td>0.19809</td>
      <td>0.28232</td>
      <td>0.28170</td>
      <td>0.28308</td>
    </tr>
    <tr>
      <td><strong>13</strong></td>
      <td>0.18178</td>
      <td>0.17980</td>
      <td>0.18050</td>
      <td>0.34110</td>
      <td>0.35198</td>
      <td>0.34029</td>
      <td>0.53839</td>
      <td>0.53612</td>
      <td>0.56625</td>
      <td>0.85703</td>
      <td>0.80011</td>
      <td>0.79524</td>
    </tr>
    <tr>
      <td><strong>19</strong></td>
      <td>0.37230</td>
      <td>0.37086</td>
      <td>0.36871</td>
      <td>0.71525</td>
      <td>0.73242</td>
      <td>0.71511</td>
      <td>1.17697</td>
      <td>1.19541</td>
      <td>1.19739</td>
      <td>1.82805</td>
      <td>1.76718</td>
      <td>1.78026</td>
    </tr>
    <tr>
      <td><strong>25</strong></td>
      <td>0.644519</td>
      <td>0.669077</td>
      <td>0.64559</td>
      <td>1.27399</td>
      <td>1.25046</td>
      <td>1.24989</td>
      <td>2.08386</td>
      <td>2.05873</td>
      <td>2.0587</td>
      <td>3.01603</td>
      <td>3.09853</td>
      <td>3.05301</td>
    </tr>
  </tbody>
</table>

It can be seen that the times recorded for inputs of the same size are very similar, indicating that the execution times of the operations are independent of the pixel values.
