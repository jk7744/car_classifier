import cv2
import argparse
from keras.models import load_model

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
args = vars(ap.parse_args())

# CNN model for sedan vs SUV Classification
model = load_model("cars_model.h5")

car_dict = { 0 : "sedan" , 1 : "SUV"}

## haar cascade
cascade_src = 'cars.xml'

#video_src = 'video.avi'

# creates video object
cap = cv2.VideoCapture(args["video"])

# creates cascade obect
car_cascade = cv2.CascadeClassifier(cascade_src)

# divide by 255.0
CAR_HEIGHT = 3.0 ##1.5 ## in metres
FOCAL_LENGTH = 0.105  ## in metres (105 mm)
SENSOR_HEIGHT = 0.035 ## in metres (35 mm)
IMAGE_HEIGHT = 1080 ## in pixels

count = 0


while True:
    if count % 10 != 0:
        cv2.imshow('video', img)
    
    else:
        # captures frame
        ret, img = cap.read()
        
        # check if frame is corrupted or not
        if (type(img) == type(None)):
            break
        
        # converts frame to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detects cars in the frame
        cars = car_cascade.detectMultiScale(gray, 1.1, 2)

        #IMAGE_HEIGHT = gray.shape[0] ## in pixels

        for (x,y,w,h) in cars:

            # crops cars one by one
            car = gray[y:y+h,x:x+w]
            # reshapes to (100,100)
            car_selection = cv2.resize(car,(100,100)).reshape(1,100,100,1)
            car_selection = car_selection/255.0
            # classify as SUV or Sedan
            out_pred = model.predict_classes(car_selection)

            # estimates distance
            dis = (FOCAL_LENGTH * CAR_HEIGHT * IMAGE_HEIGHT) // (h * SENSOR_HEIGHT) 
            distance = str(dis) + " m"
            # puts cars type
            cv2.putText(img, car_dict[out_pred[0]], (x+w, y+h),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
            
            # puts cars distance
            cv2.putText(img, distance, (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
            
            # draws bounding box
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        
        cv2.imshow('video', img)

    count += 1
   
    # q to exit
    key = cv2.waitKey(1) & 0xFF 
    if key == ord("q"):
        break

cv2.destroyAllWindows()