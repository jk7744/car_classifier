# Car Classifier and Distance Estimator

## Purpuose 
This porject aims to detect cars, and classify them as a sedan or an SUV type, and estimate their distance from the capturing camera in real time using OpenCV real time computer vision library in Python. This is particular interest considering the rate of loss of lives and the evolution of self-driving cars. Moerover, real time object detection and distance estimation have their unique challenges worth exploring.<br />
The system captures video and extracts its frame to detect objects, specifically cars, draw bounding boxes around them, and finally estimote thier distance from the camera using triangle similarity for the object/marker to camera distance technique. Vechicles are tracked real time in video frame using Haar features-the popularly used digits images features in object recongnition. A Haar cascade file tranied on cars was used in dectecting the object of interest, classify and draw boxex around them. 

Algorithm: 
1. Capture video
2. iterate over the video frames:
      when object detect
     3. convert the framce to grayscale 
     4. Detect cars of different sizes in the image.
     5. Draw rectangles around the car object. 
6. Distance Estimation 
    Estimate the distance using camera focal length together with the width of the bounding box. 
    
    
 couple of things in the code: 
 it was taken every 10 frames, otherwise it will be very fast. 
 The Distance formula was: 
 https://photo.stackexchange.com/questions/12434/how-do-i-calculate-the-distance-of-an-object-in-a-photo
 
 I have tired the write out my own model which is the SUV_Sedan.py but it keeps running for forever and also the system crashed. 
 
 for the CarData set. I used the Stanford car data set. This is the url for it: 
 https://ai.stanford.edu/~jkrause/cars/car_dataset.html
 
 

