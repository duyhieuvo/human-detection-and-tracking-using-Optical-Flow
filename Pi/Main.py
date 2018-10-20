from Rasp import *
import os

def main():
	#Stop the "Motion" streaming function to free the Pi camera
	#Prepare to scan for potential moving object
    os.system("sudo /etc/init.d/motion stop")
    while True:
        detecting = Rasp()
        detecting.detecting()
        streaming()

if __name__ == "__main__":
    main()