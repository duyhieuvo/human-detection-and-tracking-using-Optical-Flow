Realtime tracking and classification of moving objects in videos

The program is implemented in Python 2.7. 
Some external Python packages need to be installed before running the program: Numpy, Scikit-image, OpenCV, Pykalman, Tensorflow.
The 'Motion' software also needs to be installed on Raspberry Pi for streaming the video to the center computer.

To run the program:
1. Run the file Main.py of the central computer.
2. Wait until the pre-processing step is finished and the word "Begin" is printed out on the terminal.
3. Run the file Main.py of the Raspberry Pi.

To view the result:
1. Run the file GUI.py .
2. Browse to the folder of the desired object or human.
