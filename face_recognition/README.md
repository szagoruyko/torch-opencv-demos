Interactive Face Recognition with GPU
===

To provide an example of how OpenCV's CUDA module can be used in Torch, we have implemented interactive face recognition in this application. 

## Important warning

In this very example, the recognition quality may be rather poor as, in fact, the OpenFace's face recognition framework is intended to work in a more complicated way. Here, we didn't locate facial landmarks and estimate head pose, although this is an essential part of the pipeline. Keep this in mind when working with this task, and consult [OpenFace's website](http://cmusatyalab.github.io/openface/) for further information.

## How to use it

Here is how the program is run:

Usage: `th demo.lua video_source [N [Name1 Name2 ...] ]`

Where
  * *video_source*:

        Video source to capture.
        If "camera", then default camera is used.
        Otherwise, `video_source` is assumed to be a path to a video file.

  * *N*:

        Number of different people to recognize (2..9).

  * *Name1, Name2, ...*: 

        Their names (optional).

After launched, the program initializes two windows: the "gallery" (for reference face images), and the main window, which shows frames from the selected video source.

The program starts in the **learning phase**. During it, the user selects a person to be labeled from the gallery by pressing digit keys (1..9), and then double-clicks the red rectangle that contains the corresponding face. The stream can be paused with Space key for convenience. The system is ready to learn when the gallery shows no vertical red line.

![screenshot](https://cloud.githubusercontent.com/assets/9570420/13470424/2c5d3106-e0bd-11e5-9319-9f1dbf8c86ab.png)  
![screenshot 1](https://cloud.githubusercontent.com/assets/9570420/13470423/2c5d5064-e0bd-11e5-842c-d99157e22d6c.png)  

Hitting Enter brings the system into the **recognition phase**, where the names are predicted.

![screenshot 2](https://cloud.githubusercontent.com/assets/9570420/13530688/b1f694ac-e233-11e5-955c-df71688f472b.png)  
![screenshot 3](https://cloud.githubusercontent.com/assets/9570420/13530687/b1ceebd2-e233-11e5-8947-06684910aeff.png)

## How recognition works

After a face has been detected, its so-called *face descriptor* (a vector of 128 numbers) is extracted by a convolutional neural network from [OpenFace project](http://cmusatyalab.github.io/openface/). After the user has ended filling the gallery, an SVM is trained on these vectors: it tries to separate the descriptors with different labels by parabolic surfaces in 128-dimensional space. Afterwards, during the recognition phase, new descriptors (without labels, obviously) that are extracted from face detection boxes are fed to this SVM to predict their labels.