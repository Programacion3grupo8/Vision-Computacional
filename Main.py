import cv2
#capturamos video
#cap = cv2.VideoCapture(0)
#Ingresamos el algoritmo
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:
 # ret,frame = cap.read()
  image = cv2.imread('oficina.jpg')
  imageAux = image.copy()
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  faces = faceClassif.detectMultiScale(gray, 1.1, 5)

  count = 0

  for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y),(x+w,y+h),(128,0,255),2)
    rostro = imageAux[y:y+h,x:x+w]
    rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('rostro_{}.jpg'.format(count),rostro)
    count = count + 1

    cv2.imshow('rostro', rostro)
    cv2.imshow('image',image)
   
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
#cap.release()
cv2.destroyAllWindows()


