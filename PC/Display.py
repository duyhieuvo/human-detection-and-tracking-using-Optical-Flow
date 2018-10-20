import numpy as np
import cv2
import matplotlib.pyplot as plt
from Kalman import *
import os

class Display():
    def __init__(self,back,traj,time,name=None):
        self.fig, self.ax = plt.subplots()
        #Read the background image and show it on the displaying window
        img = cv2.imread(back)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #Add the name of from the human recognition if available
        plt.suptitle(name,fontsize=12)

        #Read the measured trajectory
        measured = np.load(traj)
        print measured.shape

        #Run the Kalman filter on the stored trajectory file
        self.filtered = Filter(measured)
        print self.filtered.shape

        self.time = np.load(time)
        print self.time.shape

        #If the measurement file is empty, show error
        if self.filtered.shape[0] == 0:
            self.ax.text(50, 80, 'Error: input is invalid', style='italic',
                    bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

        elif not np.isnan(self.filtered).any():

            x = self.filtered[:, 0]

            y = self.filtered[:, 1]

            #Plot the trajectory on the displaying window
            self.sc = plt.scatter(x, y)

            self.annot = self.ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="w"),
                                arrowprops=dict(arrowstyle="->"))
            self.annot.set_visible(False)

            self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        else:
            self.ax.text(50, 80, 'Error: Numerical Instability', style='italic',
                    bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})



    #Function to update the time annonation of each point on the trajectory
    def update_annot(self,ind):
        pos = self.sc.get_offsets()[ind["ind"][0]]
        self.annot.xy = pos

        text = "{}".format(" ".join([self.time[ind["ind"][0], 0]]))

        self.annot.set_text(text)
        self.annot.get_bbox_patch().set_alpha(0.4)


    #Function for displaying the time when hovering the mouse cursor on a point on the trajectory
    def hover(self,event):
        vis = self.annot.get_visible()
        if event.inaxes == self.ax:
            cont, ind = self.sc.contains(event)
            if cont:
                self.update_annot(ind)
                self.annot.set_visible(True)
                self.fig.canvas.draw_idle()
            else:
                if vis:
                    self.annot.set_visible(False)
                    self.fig.canvas.draw_idle()



    def showing(self):
        plt.show()