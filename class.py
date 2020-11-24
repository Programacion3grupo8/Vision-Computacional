import cv2
import os
import imutils
import numpy as np

class Reconocimiento:
    dataPath = 'Rostro reconocido'
    imagePaths = None
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    face_recognizer.read('modeloEigenFace.xml')
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    
    def SetDataPath():
        if not os.path.exists(Reconocimiento.dataPath):
            os.makedirs(Reconocimiento.dataPath)

    def SetData():
        Reconocimiento.SetDataPath()
        Reconocimiento.imagePaths = os.listdir(Reconocimiento.dataPath)
        Reconocimiento.face_recognizer = cv2.face.EigenFaceRecognizer_create()
        Reconocimiento.face_recognizer.read('modeloEigenFace.xml')
        Reconocimiento.faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

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
            cv2.imshow('Guardando rostro',frame)
            k =  cv2.waitKey(1)
            if k == 27 or count >=100:
                break
        cv2.destroyWindow('Guardando rostro')       

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
        Reconocimiento.SetData()
        imagePaths = os.listdir(Reconocimiento.dataPath)
        face_recognizer = cv2.face.EigenFaceRecognizer_create()
        face_recognizer.read('modeloEigenFace.xml')
        faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        while True:
            ret,frame = cap.read()
            if ret == False: break
            frame = imutils.resize(frame,width=900) 	
            frame = cv2.flip(frame,1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            auxFrame = gray.copy()

            faces = Reconocimiento.faceClassif.detectMultiScale(gray,1.3,5)
            cv2.putText(frame,'Presione G para reconocer gestos',(0, 610),2,0.8,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(frame,'Presione S para guardar rostro',(0, 640),2,0.8,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(frame,'Presione ESC para salir',(0, 670),2,0.8,(0,0,0),1,cv2.LINE_AA)

            for (x,y,w,h) in faces:
                rostro = auxFrame[y:y+h,x:x+w]
                rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
                result = Reconocimiento.face_recognizer.predict(rostro)
                cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
                
                # EigenFaces
                if result[1] < 5700:
                    cv2.putText(frame,'Rostro conocido',(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                else:
                    cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                
            cv2.imshow('Reconomiento facial',frame)
            k = cv2.waitKey(1)
            if k == 27:
                break
            elif k == ord('s'):
                Reconocimiento.GuardarRostro(cap)
                Reconocimiento.EntrenarRF()
                Reconocimiento.SetData()
            elif k == ord('g'):
                Reconocimiento.Gestos(cap)

    def Gestos(cap):
        bg = None

        #Ingresamos el algoritmo
        # faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        color_contorno = (0,255,0)
        color_ymin = (0,130,255) # Punto más alto del contorno
        color_fingers = (0,255,255)

        while True:
            ret, frame = cap.read()
            if ret == False: break

            # Redimensionar la imagen para que tenga un ancho de 640
            frame = imutils.resize(frame,width=900) 	
            frame = cv2.flip(frame,1)
            frameAux = frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = faceClassif.detectMultiScale(gray, 1.3, 5)

            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

            # Detectando dedos
            if bg is not None:

                # Determinar la región de interés
                ROI = frame[200:600,460:800]
                cv2.rectangle(frame,(460-2,200-2),(800+2,600+2),color_fingers,1)
                grayROI = cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)

                # Región de interés del fondo de la imagen
                bgROI = bg[200:600,460:800]

                # Determinar la imagen binaria (background vs foreground)
                dif = cv2.absdiff(grayROI, bgROI)
                _, th = cv2.threshold(dif, 30, 255, cv2.THRESH_BINARY)
                th = cv2.medianBlur(th, 7)
                
                # Encontrando los contornos de la imagen binaria
                cnts, _ = cv2.findContours(th,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = sorted(cnts,key=cv2.contourArea,reverse=True)[:1]

                for cnt in cnts:

                    # Encontrar el centro del contorno
                    M = cv2.moments(cnt)
                    if M["m00"] == 0: M["m00"]=1
                    x = int(M["m10"]/M["m00"])
                    y = int(M["m01"]/M["m00"])
                    cv2.circle(ROI,tuple([x,y]),5,(0,255,0),-1)

                    # Punto más alto del contorno
                    ymin = cnt.min(axis=1)
                    cv2.circle(ROI,tuple(ymin[0]),5,color_ymin,-1)

                    # Contorno encontrado a través de cv2.convexHull
                    hull1 = cv2.convexHull(cnt)
                    cv2.drawContours(ROI,[hull1],0,color_contorno,2)

                    # Defectos convexos
                    hull2 = cv2.convexHull(cnt,returnPoints=False)
                    defects = cv2.convexityDefects(cnt,hull2)
                    
                    # Seguimos con la condición si es que existen defectos convexos
                    if defects is not None:

                        inicio = [] # Contenedor en donde se almacenarán los puntos iniciales de los defectos convexos
                        fin = [] # Contenedor en donde se almacenarán los puntos finales de los defectos convexos
                        fingers = 0 # Contador para el número de dedos levantados

                        for i in range(defects.shape[0]):
                
                            s,e,f,d = defects[i,0]
                            start = cnt[s][0]
                            end = cnt[e][0]
                            far = cnt[f][0]

                            # Encontrar el triángulo asociado a cada defecto convexo para determinar ángulo					
                            a = np.linalg.norm(far-end)
                            b = np.linalg.norm(far-start)
                            c = np.linalg.norm(start-end)
                            
                            angulo = np.arccos((np.power(a,2)+np.power(b,2)-np.power(c,2))/(2*a*b))
                            angulo = np.degrees(angulo)
                            angulo = int(angulo)
                            
                            # Se descartarán los defectos convexos encontrados de acuerdo a la distnacia
                            # entre los puntos inicial, final y más alelago, por el ángulo y d
                            if np.linalg.norm(start-end) > 20 and angulo < 90 and d > 12000:
                                
                                # Almacenamos todos los puntos iniciales y finales que han sido
                                # obtenidos
                                inicio.append(start)
                                fin.append(end)
                                
                                # Visualización de distintos datos obtenidos
                                #cv2.putText(ROI,'{}'.format(angulo),tuple(far), 1, 1.5,color_angulo,2,cv2.LINE_AA)
                                #cv2.putText(ROI,'{}'.format(d),tuple(far), 1, 1.1,color_d,1,cv2.LINE_AA)
                                # cv2.circle(ROI,tuple(start),5,color_start,2)
                                # cv2.circle(ROI,tuple(end),5,color_end,2)
                                # cv2.circle(ROI,tuple(far),7,color_far,-1)
                                #cv2.line(ROI,tuple(start),tuple(far),color_start_far,2)
                                #cv2.line(ROI,tuple(far),tuple(end),color_far_end,2)
                                #cv2.line(ROI,tuple(start),tuple(end),color_start_end,2)

                        # Si no se han almacenado puntos de inicio (o fin), puede tratarse de
                        # 0 dedos levantados o 1 dedo levantado
                        if len(inicio)==0:
                            minY = np.linalg.norm(ymin[0]-[x,y])
                            if minY >= 110:
                                fingers = fingers +1
                                # cv2.putText(ROI,'{}'.format(fingers),tuple(ymin[0]), 1, 1.7,(color_fingers),1,cv2.LINE_AA)
                            
                        # Si se han almacenado puntos de inicio, se contará el número de dedos levantados
                        for i in range(len(inicio)):
                            fingers = fingers + 1
                            # cv2.putText(ROI,'{}'.format(fingers),tuple(inicio[i]), 1, 1.7,(color_fingers),1,cv2.LINE_AA)
                            if i == len(inicio)-1:
                                fingers = fingers + 1
                                # cv2.putText(ROI,'{}'.format(fingers),tuple(fin[i]), 1, 1.7,(color_fingers),1,cv2.LINE_AA)
                        
                        # Se visualiza el número de dedos levantados en el rectángulo izquierdo
                        if fingers >= 0: 
                            
                            if fingers == 0:
                                cv2.putText(frame,'{} Alerta de violencia! Llamando 911...'.format(fingers),(0,45), 1, 2,(color_fingers),2,cv2.LINE_AA)
                            elif fingers == 1:
                                cv2.putText(frame,'{} Alerta de acoso!'.format(fingers),(0,45), 1, 2,(color_fingers),2,cv2.LINE_AA)
                            elif fingers == 2:
                                cv2.putText(frame,'{} Alerta de violencia domestica!'.format(fingers),(0,45), 1, 2,(color_fingers),2,cv2.LINE_AA)
                            elif fingers == 3:
                                cv2.putText(frame,'{} Alerta de incendio! Llamando a los bomberos...'.format(fingers),(0,45), 1, 2,(color_fingers),2,cv2.LINE_AA)
                            elif fingers == 4:
                                cv2.putText(frame,'{} Alerta! Necesita primeros auxilios'.format(fingers),(0,45), 1, 2,(color_fingers),2,cv2.LINE_AA)
                            elif fingers == 5:
                                cv2.putText(frame,'{} Alerta de inundacion'.format(fingers),(0,45), 1, 2,(color_fingers),2,cv2.LINE_AA)
                            else:
                                cv2.putText(frame,'{} dedos levantados'.format(fingers),(0,45), 1, 2,(color_fingers),2,cv2.LINE_AA)
                        
                # cv2.imshow('th',th)
            
            cv2.imshow('Gestos',frame)
            bg = cv2.cvtColor(frameAux,cv2.COLOR_BGR2GRAY)

            k = cv2.waitKey(20)
            if k == 27:
                break
        cv2.destroyWindow('Gestos')


cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
Reconocimiento.Reconocer(cap)
    
cap.release()
cv2.destroyAllWindows()
    