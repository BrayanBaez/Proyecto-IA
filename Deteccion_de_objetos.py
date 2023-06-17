# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 22:17:35 2023

@author: braya
"""

import numpy as np
import cv2

# Cargar la imagen y la plantilla
img = cv2.imread('mario.png')
template = cv2.imread('gumba.png')

# Convertir la imagen y la plantilla a escala de grises
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Obtener las dimensiones de la imagen y la plantilla
imgA, imganch = gray_img.shape
templateA, templateanch = gray_template.shape

# Crear una matriz para almacenar los resultados de la correlación normalizada
corr_norm = np.zeros((imgA - templateA, imganch - templateanch))

# Calcular la media y la desviación estándar de la plantilla
template_mean = np.mean(gray_template)
template_std = np.std(gray_template)

# Realizar la correlación normalizada
for i in range(imgA - templateA):
    for j in range(imganch - templateanch):
        img_patch = gray_img[i:i+templateA, j:j+templateanch]
        img_patch_mean = np.mean(img_patch)
        img_patch_std = np.std(img_patch)
        corr_norm[i,j] = np.sum((img_patch - img_patch_mean) * (gray_template - template_mean)) / (img_patch_std * template_std * templateA * templateanch)

# Encontrar las coordenadas del máximo valor de la correlación normalizada
y,x = np.unravel_index(np.argmax(corr_norm), corr_norm.shape)

# Dibujar un rectángulo alrededor de la ubicación de la plantilla encontrada
cv2.rectangle(img, (x,y), (x+templateanch, y+templateA), (0,255,0), 2)

# Mostrar la imagen con el rectángulo dibujado
cv2.imshow('Deteccion de enemigo', img)
cv2.waitKey(0)


# Crear una matriz para almacenar los resultados de la diferencia al cuadrado
diff_sq = np.zeros((imgA - templateA, imganch - templateanch))

# Calcular la suma de las diferencias al cuadrado
for i in range(imgA - templateA):
    for j in range(imganch - templateanch):
        img_patch = gray_img[i:i+templateA, j:j+templateanch]
        diff_sq[i,j] = np.sum(np.square(img_patch - gray_template))

# Encontrar las coordenadas del mínimo valor de la diferencia al cuadrado
y,x = np.unravel_index(np.argmin(diff_sq), diff_sq.shape)

# Dibujar un rectángulo alrededor de la ubicación de la plantilla encontrada
cv2.rectangle(img, (x,y), (x+templateanch, y+templateA), (0,255,0), 2)

# Mostrar la imagen con el rectángulo dibujado
cv2.imshow('Resultado', img)
cv2.waitKey(0)
