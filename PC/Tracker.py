import numpy as np
import cv2
from datetime import datetime
import os





class Tracker():
    def __init__(self, init_bbox,frame,key,folder,background,regular):
        #The image captured when the object is first detected and will be used as a background for displaying the trajectory of the object.
        self.background = background
        # The folder directory of the connection section
        self.folder = folder
        # The unique key for this tracker
        self.key = key

        ############

        #Take the newly detected bounding box to create new tracker
        self.init_bbox = (init_bbox[0],init_bbox[1],init_bbox[2]-init_bbox[0]+1,init_bbox[3]-init_bbox[1]+1)
        #Create Median Flow tracker
        self.tracker = cv2.TrackerMedianFlow_create()
        #Initialize the tracker with the detected bounding box
        ok = self.tracker.init(frame, self.init_bbox)
        #Check if the tracking of this tracker is complete
        self.end = False
        #Find the centroid of the assigned detected bounding box
        self.centroid = np.array([[np.float32((init_bbox[0] + init_bbox[2])/2)],[np.float32((init_bbox[1] +init_bbox[3])/2)]])

        ############

        #Total frames since the tracker was created
        self.age = 1
        #Total frames that the tracked object is visible
        self.totalVisibleCount = 1
        self.consecutiveInvisibleCount =0
        #This will be switched to True to display the tracker on the video stream if the tracked object is determined to not be noise.
        self.visible = False
        #This is set to True whenever the tracker is assigned a newly detected bounding box
        self.assigned = True
        #Counter to reset the reference of the tracker after every 20 frames
        self.resetCounter = 0
        # This will be set to True if the object is marked as noise
        self.noise = False

        ############

        #Initialize the trajectory and time of recording
        self.measuredTrack = []
        self.time=[]

        ############

        # Counter for regular running of Viola Jones face detector every 20 frames
        self.detectHuman = 0
        # The number of human face detected for this object, it is also used to name the cropped face image
        self.humanFound = 0
        #The accumulated ratio of height to width for the preliminary human classifier
        self.ratio = np.float(init_bbox[3]-init_bbox[1]+1)/np.float(init_bbox[2]-init_bbox[0]+1)
        #This will be set to True if the object is marked as human
        self.human = False
        self.firstDetected = True  # check if first detected human to create a sub directory to store the image
        self.face_cascade = cv2.CascadeClassifier(
            'D:\Programs\Python\OpenCV-3.4.0\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
        # Regular face detect
        self.regular = regular


    #Tracking function of the tracker
    def tracking(self,frame,drawing,gray):
        self.assigned = False
        self.age+=1
        visibility = np.float(self.consecutiveInvisibleCount) / np.float(self.age)
        if self.consecutiveInvisibleCount>=100: #The case when the object is perceived as leaving the region of interest
            print "Consecutive Invisible"
            print "Tracker"
            print self.key
            self.delete()
        elif self.age < 20 and visibility>0.6: #The case when the object is marked as noise
            print "Noise"
            print "Tracker"
            print self.key
            self.noise = True
            self.delete()
        else:
            self.ok, self.bbox = self.tracker.update(frame) #Tracking the object in the current frame
            if self.ok: #Tracking is successfully
                self.centroid = np.array([np.float32((self.bbox[0] + self.bbox[0] + self.bbox[2])/2),np.float32((self.bbox[1] + self.bbox[1] + self.bbox[3])/2)])
                self.measuredTrack.append(([self.centroid[0], self.centroid[1]])) #Store the centroid coordination of the bounding box
                self.time.append(datetime.now().strftime('%H:%M:%S')) #Store the corresponding current time

            else: #Tracking is failed in this frame
                self.measuredTrack.append(([-1, -1]))
                self.time.append(datetime.now().strftime('%H:%M:%S'))

            if self.visible:
                if self.ok:
                    self.detectHuman +=1
                    p1 = (int(self.bbox[0]), int(self.bbox[1]))
                    p2 = (int(self.bbox[0] + self.bbox[2]), int(self.bbox[1] + self.bbox[3]))

                    #Display the tracker on the video stream
                    cv2.rectangle(drawing, p1, p2, (255, 0, 255), 2, 1)
                    cv2.putText(drawing, str(self.key), (int(self.bbox[0]), int(self.bbox[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (255, 255, 255), 2, cv2.LINE_AA)



                    ######
                    #If the preliminary human classifier is classified
                    if 3.2358 <= self.ratio <= 5.2358:
                        yy1 =  max(int(self.bbox[1]),0)
                        yy2 =  min(int(self.bbox[1] + self.bbox[3]),480)

                        xx1 =  max(int(self.bbox[0]),0)
                        xx2 =  min(int(self.bbox[0] + self.bbox[2]),640)
                        #Take the grayscale of the image patch inside the bounding box of the tracker to run the Viola Jones detector
                        gray = gray[yy1:yy2,xx1:xx2]
                        frame1 = frame[yy1:yy2,xx1:xx2]
                        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                        #detectMultiScale(image,scaleFactors,minNeighbors)

                        if len(faces)!=0:
                            cv2.putText(drawing, "Human", (int(self.bbox[0]), int(self.bbox[1] + self.bbox[3])),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2,
                                        (255, 255, 255), 2, cv2.LINE_AA)
                            if self.firstDetected:
                                self.firstDetected = False
                                os.mkdir(self.folder + "/Human" +  str(self.key) ) #Create the folder to save the cropped face images
                            self.human = True #The object is marked as human
                            #Save the human face images inside the created folder
                            for (x, y, w, h) in faces:
                                cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
                                cv2.imwrite(self.folder + "/Human" + str(self.key) + "/" +  str(self.humanFound) + ".png", cv2.resize(frame1[y:y+h,x:x+w],(182,182)))
                                self.humanFound += 1
                            cv2.imshow("Gray bbox", gray)
                    elif self.regular:
                        if self.detectHuman>=20: #Regular running the Viola Jones regardless of the result of the prilimnary classifier
                            print "Regular face detection"
                            self.detectHuman = 0
                            yy1 = max(int(self.bbox[1]), 0)
                            yy2 = min(int(self.bbox[1] + self.bbox[3]), 480)

                            xx1 = max(int(self.bbox[0]), 0)
                            xx2 = min(int(self.bbox[0] + self.bbox[2]), 640)
                            gray = gray[yy1:yy2, xx1:xx2]
                            frame1 = frame[yy1:yy2, xx1:xx2]

                            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                            if len(faces) != 0:
                                if self.firstDetected:
                                    self.firstDetected = False
                                    os.mkdir(self.folder + "/Human" + str(self.key))
                                self.human = True
                                for (x, y, w, h) in faces:
                                    cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
                                    cv2.imwrite(self.folder + "/Human" + str(self.key) + "/" + str(self.humanFound) + ".png",cv2.resize(frame1[y:y + h, x:x + w], (182, 182)))
                                    self.humanFound += 1

                                cv2.imshow("Gray bbox", gray)


                    ######


                else:
                    print "Tracking failed"
                    print "Tracker"
                    print self.key
                    cv2.putText(drawing, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 0, 255), 2)




    def reset(self,init_bbox,frame):
        self.assigned = True
        self.resetCounter+=1
        self.consecutiveInvisibleCount = 0
        self.totalVisibleCount+=1 #The object is marked as visible if it is assinged a newly detected bounding box
        if self.totalVisibleCount>10:
            self.visible=True

        #Update the reference of the tracker
        if self.resetCounter==20:
            self.resetCounter=0
            self.init_bbox = (init_bbox[0],init_bbox[1],init_bbox[2]-init_bbox[0]+1,init_bbox[3]-init_bbox[1]+1)
            self.tracker = cv2.TrackerMedianFlow_create()
            ok = self.tracker.init(frame, self.init_bbox)


    #The function for deleting the tracker when finishing tracking
    def delete(self):
        print "Tracker deleted"
        print "Tracker"
        print self.key
        self.end= True #Notify the main program to delete this tracker from its tracker list
        if (not self.noise) and self.visible: #Only if the object is not noise and its tracker is set to visible while tracking, its data will be stored
            saved = np.array(self.measuredTrack)
            time = np.array(self.time).reshape(-1, 1)

            if self.human:
                print "Human"
                cv2.imwrite(self.folder + "/Human" + str(self.key)  + "/Captured.png", self.background)
                np.save(self.folder + "/Human" + str(self.key) + "/Time" + str(self.key), time)
                np.save(self.folder + "/Human" + str(self.key) + "/Trajectory" + str(self.key) , saved)
            else:
                os.mkdir(self.folder + "/Object" + str(self.key))
                cv2.imwrite(self.folder + "/Object" + str(self.key) + "/Captured.png", self.background)
                np.save(self.folder + "/Object" + str(self.key) + "/Time" + str(self.key), time)
                np.save(self.folder + "/Object" + str(self.key) + "/Trajectory" + str(self.key), saved)




    def __del__(self):
        print "Delete object"
        print self.key