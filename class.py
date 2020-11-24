import cv2
import os
import imutils
import numpy as np

class Reconocimiento:
    dataPath = 'Rostro reconocido'

    def SetDataPath():
        if not os.path.exists(Reconocimiento.dataPath):
            os.makedirs(Reconocimiento.dataPath)

    def GuardarRostro(cap):
        Reconocimiento.SetDataPath()

        faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        count = 0

        while True:
            ret, frame = cap.read()
            if ret == False: break
            frame =  imutils.resize(frame, width=640)
            frame = cv2.flip(frame,1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            auxFrame = frame.copy()
            faces = faceClassif.detectMultiScale(gray,1.3,5)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,250,0),1)
                rostro = auxFrame[y:y+h,x:x+w]
                rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(Reconocimiento.dataPath + '/rotro_{}.jpg'.format(count),rostro)
                count = count + 1
            cv2.imshow('frame',frame)
            k =  cv2.waitKey(1)
            if k == 27 or count >=100:
                break       

    def EntrenarRF():
        peopleList = os.listdir(Reconocimiento.dataPath)
        labels = []
        facesData = []
        label = 0

        for nameDir in peopleList:
            personPath = Reconocimiento.dataPath + '/' + nameDir
            labels.append(label)
            facesData.append(cv2.imread(personPath,0))
            image = cv2.imread(personPath,0)
            # cv2.imshow('image',image)
            cv2.waitKey(10)
            label = label + 1

        # print('labels= ',labels)
        # print('Número de rostro 0: ',np.count_nonzero(np.array(labels)==0))
        # print('Número de rostro 1: ',np.count_nonzero(np.array(labels)==1))

        face_recognizer= cv2.face.EigenFaceRecognizer_create()
        # print("Creando archivo...")
        face_recognizer.train(facesData, np.array(labels))
        face_recognizer.write('modeloEigenFace.xml')

    def Reconocer(cap):
        Reconocimiento.SetDataPath()
        imagePaths = os.listdir(Reconocimiento.dataPath)
        face_recognizer = cv2.face.EigenFaceRecognizer_create()
        face_recognizer.read('modeloEigenFace.xml')
        faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        while True:
            ret,frame = cap.read()
            frame = cv2.flip(frame,1)
            if ret == False: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            auxFrame = gray.copy()
            faces = faceClassif.detectMultiScale(gray,1.3,5)
            cv2.putText(frame,'Presione G para reconocer gestos',(0, 410),2,0.8,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(frame,'Presione S para guardar rostro',(0, 440),2,0.8,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(frame,'Presione ESC para salir',(0, 470),2,0.8,(0,0,0),1,cv2.LINE_AA)

            for (x,y,w,h) in faces:
                rostro = auxFrame[y:y+h,x:x+w]
                rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
                result = face_recognizer.predict(rostro)
                cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
                
                # EigenFaces
                if result[1] < 5700:
                    cv2.putText(frame,'Rostro conocido',(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                else:
                    cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                
            cv2.imshow('frame',frame)
            k = cv2.waitKey(1)
            if k == 27:
                break
            elif k == ord('s'):
                Reconocimiento.GuardarRostro(cap)
                Reconocimiento.EntrenarRF()

    def Gestos(cap):
        print('Gestos')

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
Reconocimiento.Reconocer(cap)
    