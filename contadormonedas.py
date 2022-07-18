import cv2
import numpy as np

valorGauss=1
valorKernel=19
original=cv2.imread('monedas_eu.jpg')
gris=cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)

#Usando modelado de Gauss.
gauss=cv2.GaussianBlur(gris, (valorGauss, valorGauss), 0)

#Usando modelado de Canny.
canny=cv2.Canny(gauss, 60, 100)

#Usando modelado de kernel y matrices.
kernel=np.ones((valorKernel),np.uint8)
cierre=cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

contornos,jerarquía=cv2.findContours(cierre.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("Monedas encontradas: {}".format(len(contornos)))
cv2.drawContours(original, contornos, -1,(0,0,255),2)
#Mostrar resultados.
cv2.imshow("Grises",gris)
cv2.imshow("Gauss",gauss)
cv2.imshow("Canny",canny)
cv2.imshow("Resultado",original)
cv2.waitKey(0)