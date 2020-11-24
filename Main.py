from cv2 import cv2
import numpy as np
import imutils
# Capturando video
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# cap = cv2.VideoCapture('20201121_174929.mp4')
# cap = cv2.VideoCapture(0)
bg = None

#Ingresamos el algoritmo
faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# COLORES PARA VISUALIZACIÓN
color_start = (204,204,0)
color_end = (204,0,204)
color_far = (255,0,0)

color_start_far = (204,204,0)
color_far_end = (204,0,204)
color_start_end = (0,255,255)

color_contorno = (0,255,0)
color_ymin = (0,130,255) # Punto más alto del contorno
#color_angulo = (0,255,255)
#color_d = (0,255,255)
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
			
	cv2.putText(frame,'Presione G para reconocer gestos; Q para quitar gestos',(0, 630), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA, False)
	cv2.putText(frame,'Presione ESC para salir',(0, 660), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA, False)

	k = cv2.waitKey(20)
	if k == ord('g'):
		bg = cv2.cvtColor(frameAux,cv2.COLOR_BGR2GRAY)
	if k == ord('q'):
		bg = None
	if k == 27:
		break
	cv2.imshow('Frame',frame)
cap.release()
cv2.destroyAllWindows()