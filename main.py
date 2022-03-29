from scipy.misc import face
from store_data import Store
from face_rec import SimpleFacerec
import FDmodule
import pickle
import cv2
import os
import glob
import time

def PickleFunc(pickleF, faceToAdd, faceAdded):
    with open(pickleF, 'rb') as f:
            try:
                dictFaces = pickle.load(f)
                dictFaces[faceToAdd] = faceAdded
                with open(pickleF, 'wb') as f:
                    pickle.dump(dictFaces, f)
            except EOFError:
                dictFaces = {}
                dictFaces[faceToAdd] = faceAdded
                with open(pickleF, 'wb') as f:
                    pickle.dump(dictFaces, f)

def StoreUser(cam):
    FD = FDmodule.FaceDetector(0.75)
    SD = Store(FD)
    facesList, faceNum = SD.Face('faces.pickle')

    imgsInDir = SD.Directory()

    print("\n[INFO] Stand in the camera's view")
    SD.StoreData(cam, faceNum, imgsInDir)

    #keep track of face
    faceToAdd = f'face{faceNum}' #update Pickle File
    facesList.append(faceToAdd)

    #load encodings
    print("[INFO] serializing encodings...")
    faceAdded, hasEncodings = SD.StoreEncodings(f'Data/face{faceNum}/')
    checkAdmin = faceAdded.isAdmin 

    if (hasEncodings):
        with open('faces.pickle', 'wb') as f:
            pickle.dump(facesList, f)

        PickleFunc('encodings.pickle', faceToAdd, faceAdded)
        if checkAdmin:
            PickleFunc('admins.pickle', faceToAdd, faceAdded)

    else:
        print("Sorry, Could Not Encode Face. Please Try Again")
        files = glob.glob(f'./Data/face{faceNum}/*.jpg', recursive=True)
        try:
            for f in files:
                os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))

def Recognize(cam):
    with open('encodings.pickle', 'rb') as f:
        dictFaces = pickle.load(f)
    sfr = SimpleFacerec(dictFaces)
    valid = 0
    start = time.time()
    stop = start + 10
    while time.time() <= stop:
        print(time.time())
        ret, frame = cam.read()
    # Detect Faces
        face_locations, face_names = sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            if name != 'Unknown':
                valid += 1
                if valid == 20:
                    print(f'Access Granted, User:{name}')
                    return
            else:
                valid = 0
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    
    print('Access Denied')

def AddUser(cam):

    return

def AddAdmin(cam):

    return


cmd = input('Enter Command: ')
cam = cv2.VideoCapture(0)
while cmd != 'Quit':
    if cmd == 'Store':
        StoreUser(cam)

    elif (cmd == 'Recognize'):
        Recognize(cam)
    
    elif (cmd == 'Add User'):
        AddUser(cam)

    cmd = input('Enter Command: ')