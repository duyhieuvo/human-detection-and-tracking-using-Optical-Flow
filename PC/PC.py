import numpy as np
import cv2
import skimage.morphology
import skimage.measure
from Tools import *
from Tracker import *
from scipy.optimize import linear_sum_assignment
import time



class PC():
    def __init__(self, source, folder, grid):
        #Establish the VideoCapture object from the video stream from the Raspberry Pi
        self.cam = cv2.VideoCapture("http://" + source +":8081/")
        time.sleep(2)

        self.folder = folder

        #Initialize the parameters of the Lukas Kanade optical flow measurement
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        #Default accepting range of optical flow vector.
        self.upper = 70
        self.lower = 6

        #Initialize the spacing of the grid of interested points
        self.step = grid
        #Read the first frame from the video stream
        self.old_frame = self.cam.read()[1]
        self.old_gray = cv2.GaussianBlur(cv2.cvtColor(self.old_frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        self.avg = np.float32(self.old_gray)

        #Create the grid of interested points
        self.p_old = self.createGrid(self.old_frame) #In the form of (x,y)

        self.grid = np.uint32(self.p_old.reshape(self.p_old.shape[0], 2))
        self.column = np.int((self.p_old[self.p_old.shape[0] - 1, 0, 0] - self.p_old[0, 0, 0]) / self.step + 1)
        self.row = np.int((self.p_old[self.p_old.shape[0] - 1, 0, 1] - self.p_old[0, 0, 1]) / self.step + 1)
        self.coords = self.grid.reshape(self.row, self.column, 2).astype(np.uint32)

        ##Waiting time when there is no tracker on the list
        self.waitingTime = 100

        ### Regular human face detect: True by default
        self.regular = True

        ### Display the optical flow estimation: False by default
        self.opt = False

    #Function to switch on/off the display of optical flow measurement
    def Optical(self,enable):
        self.opt=enable

    #Function to switch on/off the regular face detection
    def RegularDetect(self,run):
        self.regular = run

    #Set threshold values of optical flow estimation
    def setFilterLimit(self,upper,lower):
        self.upper = upper
        self.lower = lower

    #Set the waiting time before closing the connection section with the Raspberry Pi after no new moving object is detected
    def setWaitingTime(self,waitingTime):
        self.waitingTime = waitingTime

    #Function to create the grid of interested points
    def createGrid(self,frame):
        grid = []
        h, w = frame.shape[:2]
        y, x = np.mgrid[self.step / 2:h:self.step, self.step / 2:w:self.step].reshape(2, -1)
        for (y1, x1) in zip(y, x):
            grid.append([[np.float32(x1), np.float32(y1)]])
        return np.array(grid)

    #Function to estimate the background of the video stream
    def Background(self, current, average, alpha):
        cv2.accumulateWeighted(current, average, alpha)
        background = cv2.convertScaleAbs(average)
        return background

    #Function to determine the moving object using background subtractor
    def backgroundSubtractor(self,gray_frame, minarea, coeff):
        background = self.Background(gray_frame, self.avg, coeff)  # Estimated background
        # cv2.imshow("Background", background)
        frameDelta = cv2.absdiff(background, gray_frame)
        thresh = cv2.threshold(frameDelta, 15, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        # cv2.imshow("Thresh",thresh)
        (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion = []
        for c in cnts:
            if (cv2.contourArea(c) > minarea):
                x, y, w, h = cv2.boundingRect(c)
                motion.append([x, y, x + w, y + h])
        motion = np.array(motion)
        new = np.array(non_max_suppression_fast(motion, 0.05))

        return new

    # Filter out unreliable optical flow vectors.
    def filter(self,p_new, st):
        d = abs(p_new - self.p_old).reshape(-1, 2).max(-1)
        for i, d1 in enumerate(d):
            if d1 > self.upper:
                p_new[i] = self.p_old[i]
                st[i] = 0
            elif d1 < self.lower:
                p_new[i] = self.p_old[i]
                st[i] = 0

    #Function to display the optical flow estimation
    def draw(self,p_new, p_old, frame):
        lines = np.hstack([p_old, p_new]).reshape(-1, 2, 2)
        lines = np.int32(lines)
        display = frame.copy()
        cv2.polylines(display, lines, 0, (0, 255, 0))
        for (x1, y1) in self.grid:
            cv2.circle(display, (x1, y1), 2, (0, 255, 0), -1)
        cv2.imshow("Optical flow",display)

    #Function to create bounding box around moving object detected on the grid of interested points
    def groupingOpticalflow(self,st):
        group = []
        st = st.reshape(self.row, self.column)
        labeled = skimage.morphology.label(st)
        #Remove object containing fewer than 5 interested points
        labeled = skimage.morphology.remove_small_objects(labeled, 5, connectivity=2)

        props = skimage.measure.regionprops(labeled)
        for prop in props:
            pos = prop.bbox
            group.append(((self.coords[pos[0], pos[1]]), (self.coords[pos[2] - 1, pos[3] - 1])))

        group = np.array(group).reshape(-1, 4)
        return group

    # Check overlap between motion field and field from background subtractor
    def checkOverlap(self,l1_x, l1_y, r1_x, r1_y, l2_x, l2_y, r2_x, r2_y):
        if (l1_x > r2_x) or (l2_x > r1_x):
            return False
        if (l1_y > r2_y) or (l2_y > r1_y):
            return False
        return True

    #Function to determine area of a bounding box
    def Area(self,l1_x, l1_y, r1_x, r1_y):
        return (r1_x - l1_x) * (r1_y - l1_y)

    #Function to find the minimum area in a list of bounding boxes
    def MinArea(self,bbox):
        if bbox.shape[0] != 0:
            for i, box in enumerate(bbox):
                area = self.Area(box[0], box[1], box[2], box[3])
                if i == 0:
                    minarea = area
                elif (area < minarea):
                    minarea = area
            return minarea


    #Combine the optical flow motion field and field from background subtractor to generate final detection result.
    def Combination(self,background, motion):
        if background.shape[0] == 0:
            return np.array(non_max_suppression_fast(motion, 0.1))
        bbox = []
        minmotion = self.MinArea(motion)
        count = 0  # count the number of back that smaller than min area of motion, if all back in background is smaller, use motion only.
        for back in background:
            if self.Area(back[0], back[1], back[2], back[3]) < minmotion:
                count += 1
                continue
            xx1 = 0
            yy1 = 0
            xx2 = 0
            yy2 = 0
            overlap = False
            counter = 0
            for opt in motion:
                if self.checkOverlap(back[0], back[1], back[2], back[3], opt[0], opt[1], opt[2], opt[3]):
                    if counter == 0:
                        overlap = True
                        xx1 = opt[0]
                        yy1 = opt[1]
                        xx2 = opt[2]
                        yy2 = opt[3]
                    else:
                        if (opt[0] < xx1):
                            xx1 = opt[0]
                        if (opt[1] < yy1):
                            yy1 = opt[1]
                        if (opt[2] > xx2):
                            xx2 = opt[2]
                        if (opt[3] > yy2):
                            yy2 = opt[3]
                    # motion = np.delete(motion,,0)
                    counter += 1

            if (overlap):
                xx1 = np.maximum(back[0], xx1)
                yy1 = np.minimum(back[1], yy1)
                xx2 = np.minimum(back[2], xx2)
                yy2 = np.maximum(back[3], yy2)
                bbox.append((xx1, yy1, xx2, yy2))

        if count != background.shape[0]:
            new = np.array(non_max_suppression_fast(np.array(bbox), 0.1))
        else:
            new = np.array(non_max_suppression_fast(motion, 0.1))
        return new

    def detector(self,old_gray, frame_gray,frame):
        back = self.backgroundSubtractor(frame_gray, 500, 0.05)

        p_new, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, self.p_old, None, **self.lk_params)

        self.filter(p_new, st)
        if self.opt:
            self.draw(p_new[st==1],self.p_old[st==1],frame)
        group = self.groupingOpticalflow(st)
        comb = self.Combination(back, group)
        return comb

    #Find the centroids of the detected bounding box and append it to the result
    def centroid(self,combs):
        if combs.shape[0] == 0:
            return combs
        else:
            x = ((combs[:, 0] + combs[:, 2]) / 2).reshape(-1, 1)
            y = ((combs[:, 1] + combs[:, 3]) / 2).reshape(-1, 1)
            combs = np.append(combs, x, axis=1)
            return np.append(combs, y, axis=1)


    #Function to create the cost matrix
    def costMatrix(self, trackerList, combs):
        row = len(trackerList)
        col = combs.shape[0]
        dis = np.zeros((row, col), dtype=float) #Distance between centroids
        ovl = np.zeros((row, col), dtype=float) #Overlap area of the tracked and detected bounding boxes
        for i, comb in enumerate(combs):
            sum = 0
            for j, tracker in enumerate(trackerList):
                if not tracker.ok:
                    distance = np.linalg.norm(tracker.centroid - np.array([comb[4], comb[5]], dtype=float))
                    sum += distance
                    dis[j, i] = distance
                    ovl[j, i] = 0
                else:

                    distance = np.linalg.norm(tracker.centroid - np.array([comb[4], comb[5]], dtype=float))
                    sum += distance
                    tracker_bbox = np.array([tracker.bbox[0], tracker.bbox[1], tracker.bbox[0] + tracker.bbox[2],
                                             tracker.bbox[1] + tracker.bbox[3]])
                    detector_bbox = np.array([comb[0], comb[1], comb[2], comb[3]])
                    overlap = overlapArea(tracker_bbox, detector_bbox)

                    dis[j, i] = distance
                    ovl[j, i] = (1 - overlap)
            dis[:, i] /= (sum)

        cost = dis + ovl

        return cost

    def Running(self):
        trackerList = []
        key = 0
        counter = 0
        while True:
            (grabbed, frame) = self.cam.read()
            if not grabbed:
                break

            copy = frame.copy()
            frame_gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #Filter high frequency noise as a pre-processing step for optical flow estimation
            frame_gray = cv2.GaussianBlur(frame_gray1, (21, 21), 0)

            old_combs = self.detector(self.old_gray, frame_gray,frame)
            combs = self.centroid(old_combs)



            for coords in combs:
                cv2.rectangle(copy, (coords[0], coords[1]), (coords[2], coords[3]), (255, 255, 0), 2)

            for tracker in trackerList:
                tracker.tracking(frame, copy, frame_gray1)


            if (len(trackerList) > 0) and (combs.shape[0] > 0):

                cost = self.costMatrix(trackerList, combs)

                tracker_ind, detector_ind = linear_sum_assignment(cost)
                i = 0
                for indx in detector_ind:
                    #Update the accumulated height-to=width ratio of each assigned tracker
                    trackerList[i].ratio = (trackerList[i].ratio + np.float(
                        combs[indx][3] - combs[indx][1] + 1) / np.float(combs[indx][2] - combs[indx][0] + 1)) / 2
                    #Mark the assigned tracker as visible and update its reference if its counter is reset
                    trackerList[i].reset(combs[indx], frame)
                    i += 1
                combs = np.delete(combs, detector_ind, 0)



            #Mark the unassigned tracker as invisible and delete the old tracker
            for i, tracker in enumerate(trackerList):
                if not tracker.assigned:
                    tracker.consecutiveInvisibleCount += 1
                if tracker.end:
                    del trackerList[i]


            #Create new tracker from remaining detected bounding box
            for comb in combs:
                tracker = Tracker(comb, frame, key,self.folder,takeImage(self.cam),self.regular)
                key += 1
                trackerList.append(tracker)

            # print key

            # if len(trackerList)==0:
            #     counter+=1
            # else:
            #     counter = 0
            #
            # if counter == self.waitingTime:
            #     break

            cv2.imshow("Tracking", copy)


            keyword = cv2.waitKey(20) & 0xFF
            if keyword == ord("q"):
                break
            self.old_gray = frame_gray.copy()

        cv2.destroyAllWindows()
        self.cam.release()

    def __del__(self):
        print "Tracking done"