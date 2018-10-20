import socket
from PC import *
import os
from datetime import datetime
#from Recognition import *
import gc

TCP_IP = '192.168.137.1'  #The IP address of the central computer
TCP_PORT = 5005 #The port number used for sending message between central computer and Raspberry Pi
BUFFER_SIZE = 20

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #create network socket with IPv4 and TCP protocol.
s.bind((TCP_IP, TCP_PORT))
s.listen(1) #Convert to server socket with the queue of maximum 1 connection.


#recognizer = humanRecognition()

while True:
    print "Begin"
    gc.collect() #Delete remaining objects from the last connection section
    conn, addr = s.accept() #waiting for connection request from Raspberry Pi
    print ('Connection address:', addr)

    #Create folder for this connection section
    folder = addr[0]  + " " + datetime.now().strftime('%m-%d %H.%M.%S')
    os.makedirs(folder)

    tracking = PC(addr[0],folder, 20)
    #tracking.Optical(True) #Uncomment this line to display the measurement of optical flow
    #tracking.RegularDetect(False) #Uncomment this line to turn off the regular run of Viola Jones detector
    #tracking.setFilterLimit(upper cut-off value, lower cut-off value) #Use this function to set the threshold value of the estimated optical flow.
    tracking.setWaitingTime(250) #Set the maximum number of frames before terminate the connection section with the Raspberry Pi when there is no new object is detected
    tracking.Running()

    conn.send("Done")
    conn.close()
    #Run the human recognition function
    #recognizer.recognizing(folder)

    print "End"