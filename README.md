Torch7-OpenCV demos
==================

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
classification pretrained network to classify what it see on it.

Run as `th demo.lua`

# Age&Gender prediction

This demo uses two networks described here http://www.openu.ac.il/home/hassner/projects/cnn_agegender/
to predict age and gender of the faces that it finds with a simple cascade detector.

Run as
```
th demo.lua `locate haarcascade_frontalface_default.xml`
```

# Credits

2016 Sergey Zagoruyko
