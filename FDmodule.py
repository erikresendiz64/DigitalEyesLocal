import cv2
import time
import mediapipe as mp

class FaceDetector():
    def __init__(self, minConfidence = 0.5): #initialize
        self.minConfidence = minConfidence 
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minConfidence) #minimum 75% confidence

    def findFaces(self, frame, draw = True):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert to greyscale

        self.results = self.faceDetection.process(frameRGB) #process every frame

        bounds = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                boundC = detection.location_data.relative_bounding_box #data is relative to the frame size, so we need to change it
                frameh, framew, framec = frame.shape #our frame size (height, width, channel)
                bound = int(boundC.xmin * framew), int(boundC.ymin * frameh), \
                        int(boundC.width * framew), int(boundC.height * frameh) #adjust rectangle bounds, manipulate locations
                bounds.append(bound)
        
        return frame, bounds

def main():
    cam = cv2.VideoCapture(0)
    detector = FaceDetector(0.75)

    print("\n [INFO] Stand in the camera's view")
    numImgs = 0
    while True:
        ret, frame = cam.read() #read each frame, return true if a frame exists
        frame, bounds = detector.findFaces(frame)
        cv2.imshow("Running", frame)
        k = cv2.waitKey(1)
        if k % 256 == 27:
            break

        
if __name__ == "__main__":
    main()