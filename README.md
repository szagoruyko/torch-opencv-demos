Torch7-OpenCV demos
==================

Real-time demos that use deep convolutional neural networks to classify and caption
what they see in real-time from a webcam stream.

All demos use CPU, but it's trivial to fix them to work with CUDA or OpenCL.

Quick install on OS X:

```bash
brew instal opencv3 --with-contrib
OpenCV_DIR=/usr/local/Cellar/opencv3/3.1.0/share/OpenCV luarocks install cv
brew install protobuf
luarocks install loadcaffe
```

In Linux you have to build OpenCV3 manually. Follow the instructions in

* https://github.com/VisionLabs/torch-opencv
* https://github.com/szagoruyko/loadcaffe

# ImageNet classification

The demo simply takes a central crop from a webcam and uses a small ImageNet
classification pretrained network to classify what it see on it. top-5 predicted
classes are shown on top, the top one is the most probable.

Run as `th demo.lua`

Example:

![sunglasses](https://cloud.githubusercontent.com/assets/4953728/14815521/2095c832-0bac-11e6-80bc-09b19c13271d.png)

# Age&Gender prediction

This demo uses two networks described here http://www.openu.ac.il/home/hassner/projects/cnn_agegender/
to predict age and gender of the faces that it finds with a simple cascade detector.

Run as
```
th demo.lua `locate haarcascade_frontalface_default.xml`
```

IMAGINE Lab gives an example:

![age&gender](https://cloud.githubusercontent.com/assets/4953728/12299217/fc819f80-ba15-11e5-95de-653c9fda9b83.png)

NeuralTalk2 demo
================

This demo uses NeuralTalk2 captioning code from Andrej Karpathy: https://github.com/karpathy/neuraltalk2

The code captions live webcam demo. Follow the installation instructions at
https://github.com/karpathy/neuraltalk2 first and then run the demo as:

```
th videocaptioning.lua -gpuid -1 -model model_id1-501-1448236541_cpu.t7
```

Caption is displayed on top:

![neuraltalk2](https://cloud.githubusercontent.com/assets/4953728/14815525/23cfb3aa-0bac-11e6-84fd-dd0f7a33422d.png)

# Realtime stylization with texture networks

Check https://github.com/DmitryUlyanov/texture_nets

![screen shot 2016-04-25 at 00 08 15](https://cloud.githubusercontent.com/assets/4953728/14781476/fa8a7c1a-0ae2-11e6-88fb-10e2bf418d86.png)

# Credits

2016 Sergey Zagoruyko and Egor Burkov

Thanks to VisionLabs for putting up https://github.com/VisionLabs/torch-opencv bindings!
