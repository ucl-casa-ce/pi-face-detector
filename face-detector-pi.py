import cv2
import sys
import time

# The classifier file can be downloaded from https://github.com/opencv/opencv/tree/master/data/haarcascades
cascPath = "haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

i=0
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    scale_percent = 50 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # Resize image
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around the faces
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Select the region of interest
        roi = frame[y:y+h, x:x+w].copy()
        cv2.imwrite('{}.jpg'.format(i), roi)
        
    # Display the resulting frame
    cv2.imshow('Video', frame)

    i += 1
  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
