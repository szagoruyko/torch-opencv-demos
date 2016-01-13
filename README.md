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
* https://githib.com/szagoruyko/loadcaffe

# ImageNet classification

The demo simply takes a central crop from a webcam and uses a small ImageNet
classification pretrained network to classify what it see on it. top-5 predicted
classes are shown on top, the top one is the most probable.

Run as `th demo.lua`

Example:

![sunglasses](https://cloud.githubusercontent.com/assets/4953728/12299791/d984309e-ba18-11e5-9838-afcfe9cdaf79.png)

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

![neuraltalk2](https://cloud.githubusercontent.com/assets/4953728/12300267/01abc18e-ba1b-11e5-8b2b-da9c9141fd55.png)

# Credits

2016 Sergey Zagoruyko

Thanks to VisionLabs for putting up https://github.com/VisionLabs/torch-opencv bindings!
