import os
import glob
import re
from os import mkdir
import cv2
import time
import mediapipe as mp
import FDmodule
import pickle
import face_recognition

def FindNumInString(str):
    findNum = re.findall('[0-9]+', str)
    num = int(findNum[0])

    return num

class Person:
    def __init__(self):
        self.ID = 0
        self.encodings = []
        self.isAdmin = False
    
    def MakeAdmin(self):
        self.isAdmin = True

class Store:
    def __init__(self, FD):
        self.FD = FD

    def Face(self, file):
        with open('faces.pickle', 'rb') as f:
            try:
                facesList = pickle.load(f)
                lastFace = facesList[len(facesList) - 1]
                numInStr = FindNumInString(lastFace)
                self.faceNum = numInStr + 1
            except EOFError:
                facesList = []
                self.faceNum = 0
            
            return facesList, self.faceNum

    def Directory(self):
        if not os.path.exists(f'./Data/face{self.faceNum}'): #if face not yet in Data folder, 
            mkdir(f"./Data/face{self.faceNum}")
            numImgs = 0
        else:
            dir = os.listdir(f'./Data/face{self.faceNum}') #else it does exist, list directory
            if len(dir) == 0: #if there are no images
                numImgs = 0 #start at 0 images
            else:
                list_of_files = glob.glob(f'./Data/face{self.faceNum}/*jpg') # * means all if need specific format then *.csv
                latest_file = max(list_of_files, key=os.path.getctime)
                parsed = latest_file.split('/')
                numLoc = parsed[3]
                numInStr = FindNumInString(numLoc)
                numImgs = numInStr #to avoid overwriting previous images ^^^
        return numImgs

    def StoreData(self, cam, faceNum, imgsInDir):
        while True:
            ret, frame = cam.read() #read each frame, return true if a frame exists
            frame, bounds = self.FD.findFaces(frame)

            imgsInDir += 1
            if(len(bounds) != 0):
                if imgsInDir % 5 == 0:
                    cv2.imwrite(f"./Data/face{faceNum}/face{faceNum}.{int(imgsInDir/5)}.jpg", frame)
                    cv2.imwrite(f"./Dataset/face{faceNum}.{int(imgsInDir/5)}.jpg", frame)
                pass
            cv2.imshow('Running', frame)
            
            k = cv2.waitKey(1)

            if k % 256 == 27:
                break
            elif imgsInDir >= 20: 
                break

    def StoreEncodings(self, images_path):
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        print("{} encoding images found.".format(len(images_path)))
        
        newFace = Person()
        encodings = []
        img_encoded = 0
        # Store image encoding and names
        idx = 0
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            paths = img_path.split("/")
            info = paths[1]
            if info.startswith('.'):
                pass
            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            idx = FindNumInString(filename)
            face = face_recognition.face_encodings(rgb_img)
            # Get encoding
            if face:
                img_encoding = face[0]
                img_encoded += 1
                encodings.append(img_encoding)
                # encodings.append(img_encoding)
                # names.append(name)
                print("image succesfully encoded")
            else:
                print("encoding unsuccesful")
        
        if img_encoded == 0:
            hasEncodings = False
        else:
            hasEncodings = True
            newFace.ID = idx
            newFace.encodings = encodings

            if idx == 0:
                newFace.MakeAdmin()

        print(f'{img_encoded} images encoded')
        print("\nEncoding images loaded")

        return newFace, hasEncodings