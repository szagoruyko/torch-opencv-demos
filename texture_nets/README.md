Realtime texture_nets
================

This enables realtime stylization with texture networks, for more information check https://github.com/DmitryUlyanov/texture_nets

For now 'starry night' and 'pollock' models are available, we will add more in future.

![screen shot 2016-04-25 at 00 08 15](https://cloud.githubusercontent.com/assets/4953728/14781476/fa8a7c1a-0ae2-11e6-88fb-10e2bf418d86.png)

# Usage

The demo depends on folding Batch Normalization into convolution in [imagine-nn](https://github.com/szagoruyko/imagine-nn), to install it do:

```
luarocks install inn
```

To run on CPU do:

```
OMP_NUM_THREADS=2 th stylization.lua
```

On my dual core macbook it takes about ~0.6s to process one frame.

To run on CUDA do:

```
type=cuda th stylization.lua
```

OpenCL is supported to with `type=cl`. Might be slower than CPU though.

To run on input video file do:

```
video_path=*path to file* th stylization.lua
```

# Model 

![outfile](https://cloud.githubusercontent.com/assets/4953728/14781485/02f31ad8-0ae3-11e6-9cdc-8660c34384b3.png)

# Credits

Thanks to Dmitry Ulyanov for providing the initial version of this demo and the working network.

