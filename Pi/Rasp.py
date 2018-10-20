import numpy as np
import cv2
import skimage.morphology
import skimage.measure
import time
import socket
import os



def streaming():
    os.system("sudo /etc/init.d/motion start")
    TCP_IP = '192.168.137.1'  # this IP of my pc. When I want raspberry pi 2`s as a client, I replace it with its IP '169.254.54.195'
    TCP_PORT = 5005
    BUFFER_SIZE = 1024
    MESSAGE = "Streaming"

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))
    s.send(MESSAGE)
    
    while True:
        data = s.recv(BUFFER_SIZE)
        if data == "Done":
            break
    s.close()
    print ("received data:", data)
    os.system("sudo /etc/init.d/motion stop")

class Rasp():
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        time.sleep(2)

        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.step = 20

        self.old_frame = self.cam.read()[1]

        self.old_gray = cv2.cvtColor(self.old_frame, cv2.COLOR_BGR2GRAY)
        self.old_gray = cv2.GaussianBlur(self.old_gray, (21, 21), 0)

        self.p_old = self.createGrid(self.old_frame)

        self.grid = np.uint32(self.p_old.reshape(self.p_old.shape[0], 2))
        self.column = np.int((self.p_old[self.p_old.shape[0] - 1, 0, 0] - self.p_old[0, 0, 0]) / self.step + 1)
        self.row = np.int((self.p_old[self.p_old.shape[0] - 1, 0, 1] - self.p_old[0, 0, 1]) / self.step + 1)
        self.coords = self.grid.reshape(self.row, self.column, 2).astype(np.uint32)


    def createGrid(self,frame):
        grid = []
        h, w = frame.shape[:2]
        y, x = np.mgrid[self.step / 2:h:self.step, self.step / 2:w:self.step].reshape(2, -1)
        for (y1, x1) in zip(y, x):
            grid.append([[np.float32(x1), np.float32(y1)]])
        return np.array(grid)






# Filter out unreliable optical flow vectors.
    def filter(self,p_new, st):
        d = abs(p_new - self.p_old).reshape(-1, 2).max(-1)
        for i, d1 in enumerate(d):
            if d1 > 70:
                p_new[i] = self.p_old[i]
                st[i] = 0
            elif d1 < 6:
                p_new[i] = self.p_old[i]
                st[i] = 0

# Group pixels in motion together to find the moving object. No LBP is included.
    def groupingOpticalflow(self,st):
        group = []
        st = st.reshape(self.row, self.column)
        labeled = skimage.morphology.label(st)
        labeled = skimage.morphology.remove_small_objects(labeled, 5, connectivity=2)

        props = skimage.measure.regionprops(labeled)
        for prop in props:
            pos = prop.bbox
            group.append(((self.coords[pos[0], pos[1]]), (self.coords[pos[2] - 1, pos[3] - 1])))

        group = np.array(group).reshape(-1, 4)
        return group





    def detecting(self):
        time.sleep(2)
        motion = False
        counter = 0
        while not motion:
            (grabbed, frame) = self.cam.read()
            if not grabbed:
                break

            test = frame.copy()
            frame_gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.GaussianBlur(frame_gray1, (21, 21), 0)

            p_new, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p_old, None, **self.lk_params)

            self.filter(p_new, st)
            group = self.groupingOpticalflow(st)

            for coords in group:
                cv2.rectangle(test, (coords[0], coords[1]), (coords[2], coords[3]), (255, 255, 0), 2)

            cv2.imshow("Combination", test)
            if group.shape[0]!=0:
                counter+=1
            else:
                counter =0
            if counter == 6:
                motion = True

            keyword = cv2.waitKey(20) & 0xFF
            if keyword == ord("q"):
                break
            self.old_gray = frame_gray.copy()

        cv2.destroyAllWindows()
        self.cam.release()


