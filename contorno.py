import cv2
from cv2 import findContours
imagen=cv2.imread('contorno.jpg')

#Convertir imagen a colores grises.
grises=cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)


_, umbral=cv2.threshold(grises,100,255,cv2.THRESH_BINARY)
contorno, jerarquía = cv2.findContours(umbral,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imagen,contorno,-1,(251, 63, 52),3)

#Mostrar.
cv2.imshow('imagen',imagen)
#cv2.imshow('imagen a escala de grises',grises)
#cv2.imshow('imagen umbral',umbral)
cv2.waitKey(0)
cv2.destroyAllWindows()