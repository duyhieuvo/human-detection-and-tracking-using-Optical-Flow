# human-detection-and-tracking-using-Optical-Flow

This is my Bachelor Thesis Topic to implement a system that can be able to detect moving objects in a region of interest, track their movement and classify them into Human/Non-Human groups.

The detection is done using Optical Flow measurement in combination with simple background subtraction. The Median Flow tracker is then used to tracked the object. Position of the object is recorded for later visualization and inspection. The object is classified using a two-step process: Golden Ratio measurement and frontal human face detector using Viola Jones framework.

The system consists of two main parts: a Raspberry Pi 3 and a normal computer. The Pi mini-computer is in charge of scanning the region of interest to detect for any potential moving object. If an object is found, it will streamed the video captured by its camera to the central computer for in-depth analysis.
