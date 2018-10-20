import tkFileDialog as filedialog
from Tkinter import *
from Display import *
import os

def browse_button():
    # Allow user to select a directory and store it in global variable folder_path
    global folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    #List all files in the folder
    files = os.listdir(filename)
    print files
    name = ""
    #Loop through all files in the folder
    for i,file in enumerate(files):
        print file
        #Find the "Trajectory" file
        if "Trajectory" in file:
            trajectory = i
            print trajectory
        #Find the "Time" file
        elif "Time" in file:
            time = i
            print time
        #Find the background image for displaying the result
        elif "Captured" in file:
            background = i
            print background
        #Find file with the result of human recognition if available
        elif file.endswith(".txt"):
            name = os.path.splitext(file)[0]
            print name
    print files[background]
    if name!="":
        showing = Display(filename + "/" + files[background],filename + "/" +  files[trajectory],filename + "/" +  files[time],name)
    else:
        showing = Display(filename + "/" + files[background],filename + "/" +  files[trajectory],filename + "/" +  files[time])
    showing.showing()


    print(filename)


root = Tk()
folder_path = StringVar()
lbl1 = Label(master=root,textvariable=folder_path)
lbl1.grid(row=0, column=1)
button2 = Button(text="Browse", command=browse_button)
button2.grid(row=0, column=3)

mainloop()