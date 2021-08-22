import tensorflow as tf 
import cv2
import numpy as np
model = tf.keras.models.load_model('./new_mobinet_model.h5')
import cv2
import numpy as np
import winsound
frequency = 2500 #setting the frquency to 2500 hertz
duration = 1000  #setting the duration to 1000ms which is equivalent to 1 sec

cap = cv2.VideoCapture(0)
#checking if the webcam is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("Cannot open the webcam")
    
counter = 0
# font
font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 2
while True:
    ret,frame = cap.read()    
   
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
     
    img = cv2.resize(frame,(224, 224))
    final_image = img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    final_image = np.expand_dims(final_image,axis = 0)
    final_image = final_image/255.0
    
    
    predictions = model.predict(final_image)
    if(predictions <= 0.5):
        status ="Active"
        print("Active: ", predictions)
        cv2.putText(frame,status,org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    else:
        counter = counter +1
        if counter>5:
            status = "Drowsy"
            print("Drowsy: ", predictions)
            cv2.putText(frame,status,org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
            winsound.Beep(frequency,duration)
            counter = 0

    cv2.imshow('Drowsiness detection',frame)
    if cv2.waitKey(2) &0xFF == ord('q'):
        break
    
    
cap.release()
cv2.destroyAllWindows()