import matplotlib.pyplot as plt
import cv2


face_cascade = cv2.CascadeClassifier('/Users/santiagoyeomans/Developer/Face Detection/haarcascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)

while True:

    ret,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x,y,w,h in faces:

        cv2.rectangle(gray, (x,y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow('Face Detection', gray)
    plt.show()

    if cv2.waitKey(1) == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows()