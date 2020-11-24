import cv2
import os
import numpy as np

dataPath =  'C:/Users/elpip/Desktop/Reconocimiento/Data'
peopleList = os.listdir(dataPath)
print('Personas reconocidas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
	personPath = dataPath + '/' + nameDir
	print('Leyendo las imágenes')

	for fileName in os.listdir(personPath):
		print('Rostros: ', nameDir + '/' + fileName)
		labels.append(label)
		facesData.append(cv2.imread(personPath+'/'+fileName,0))
		image = cv2.imread(personPath+'/'+fileName,0)
		cv2.imshow('image',image)
		cv2.waitKey(10)
	label = label + 1

print('labels= ',labels)
print('Número de rostro 0: ',np.count_nonzero(np.array(labels)==0))
print('Número de rostro 1: ',np.count_nonzero(np.array(labels)==1))


face_recognizer= cv2.face.EigenFaceRecognizer_create()


print("Creando archivo...")

face_recognizer.train(facesData, np.array(labels))

face_recognizer.write('modeloEigenFace.xml')

