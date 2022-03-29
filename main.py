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

def StoreUser(cam, updatingAdmin):
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
    if updatingAdmin:
        faceAdded.isAdmin = True
    else:
        pass
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

def Recognize(cam, checkAdmin):
    if checkAdmin:
        with open('admins.pickle', 'rb') as f:
            dictFaces = pickle.load(f)
    else:
        with open('encodings.pickle', 'rb') as f:
            dictFaces = pickle.load(f)

    sfr = SimpleFacerec(dictFaces)
    valid = 0
    start = time.time()
    stop = start + 10
    while time.time() <= stop:
        ret, frame = cam.read()
    # Detect Faces
        face_locations, face_names = sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            if name != 'Unknown':
                valid += 1
                if valid == 20:
                    return True, name
            else:
                valid = 0
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    
    return False, name

def AddUser(cam):
    adminRec, name = Recognize(cam, checkAdmin=True)

    return adminRec, name

def AddAdmin(cam):
    adminRec, name = Recognize(cam, checkAdmin=True)
    
    return adminRec, name

def Remove(user):
    userFind = f'face{user}'
    with open('encodings.pickle') as f:
        users = pickle.load(f)
        if userFind in users:
            del users[userFind]
            with open('encodings.pickle', 'wb') as f:
                pickle.dump(users,f)
        else:
            print('User does not exist. Make sure user is correct')
    

cmd = input('Enter Command: ')
cam = cv2.VideoCapture(0)
while cmd != 'Quit':
    if cmd == 'Setup':
        StoreUser(cam, updatingAdmin=True)

    elif (cmd == 'Recognize'):
        isUser, user = Recognize(cam, checkAdmin=False)
        if isUser:
            print(f'Access Granted, User{user}.')
        else:
            print(f'Access Denied')
    
    elif (cmd == 'Add User'):
        isAdmin, user = AddUser(cam)
        if isAdmin:
            print(f'Admin Request Allowed, user: {user}')
            print('New User, Get Ready To Be Scanned')
            time.sleep(10)
            StoreUser(cam, updatingAdmin=False)
        else:
            print(f'Sorry, Admin Access Only')
    
    elif (cmd == 'Add Admin'):
        isAdmin, user = AddUser(cam)
        if isAdmin:
            print(f'Admin Request Allowed, user{user}')
            print('New Admin, Get Ready To Be Scanned')
            time.sleep(10)
            isUser, user = Recognize(cam, checkAdmin=False)
            if isUser:
                print('Recognized user')
                with open('encodings.pickle', 'rb') as f:
                    faceToAdd = f'face{user}'
                    userFaces = pickle.load(f)
                    userFaces[faceToAdd].isAdmin = True #double check it updates inside the pickle file
                
                updatedUser = userFaces[faceToAdd]
                with open('admins.pickle', 'rb') as f:
                    admins = pickle.load(f)
                    admins[faceToAdd] = updatedUser
                    with open('admins.pickle', 'wb') as f:
                        pickle.dump(admins, f)
                    
            else:
                StoreUser(cam, updatingAdmin=True)
        
        else:
            print(f'Sorry, Admin Access Only')

    cmd = input('Enter Command: ')