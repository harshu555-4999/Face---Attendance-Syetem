from inspect import _void
from posixpath import curdir
from pydoc import classname
from sre_constants import SUCCESS
from urllib.request import AbstractDigestAuthHandler
import face_recognition
import cv2
import numpy as np
import csv
import os
import os.path
import glob
from datetime import datetime


source = 0

now = datetime.now()
current_date = now.strftime("%d-%m-%Y") # Date format   dd-mm-yyyy

path = 'ImagesAttendence'
currentfilepath = current_date+'.csv'
images = []
classNames = []
classID = []  ####
myList = os.listdir(path)
#print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    classID.append(os.path.splitext(cl)[1])
#print(classNames)
#print("class ID = ",classID)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


def markAttendence(name):
    if os.path.exists(currentfilepath):
        with open(currentfilepath,'r+') as f:
            myDataList = f.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList: 
                #print("name not in list")
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')
                #print("name just added")
            #else:
                #print("name already added")
    else:
        #print("No file detected")
        with open(currentfilepath,'w') as f:
            pass

encodeListKnown = findEncodings(images)
print("Encoding Complete")
print("web cam Opening")

cap = cv2.VideoCapture(source)


while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendence(name)


    cv2.imshow('Webcam',img)
    cv2.waitKey(1)

