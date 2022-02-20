from ctypes import resize
from PIL import Image, ImageOps
import PIL
import glob
import AMPEdnn as dnn
import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
import os

n = 10  # el numero de imagenes
WEIGHTS_PATH = './my_weights.pt'

def negativo(img):  # función que simplemente invierte el valor del color de cada pixel de la imagen
    w,h=img.size
    for i in range(w):
        for j in range(h):
            color=img.getpixel((i,j))
            color=255-color # se invierte el gris
            img.putpixel((i,j),color)
            

def main():
    
    # se define la transformación a aplicar a las imagenes para convertirlas en tensores
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),])
    # lee las fotos del directorio imagenes, las redimensiona y las deja en escala de grises
    imagenes = []
    for i in range(n):
        aux = Image.open("imagenes/" + str(i) + ".jpeg")
        img = ImageOps.grayscale(aux.resize((28,28)))
        negativo(img)   # es necesario aplicarle este filtro de negativo porque si no la dnn no reconoce los 
                        # dígitos, ya que resultan imágenes de blanco sobre negro en vez de negro sobre blanco
        tensor = transform(img) # se aplica la transformación
        imagenes.append(tensor)
            
    # se crea el model de DNN a partir del fichero AMPEdnn.py
    model = dnn.My_DNN()
    model.load_state_dict(torch.load(WEIGHTS_PATH))
   
   # se recorren las 10 fotos
    for i in range(10):
        img = imagenes[i].view(1, 784)
        with torch.no_grad():
            logps = model(img)
            
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        print("\nPredicted Digit =", probab.index(max(probab)))
        print("True digit=", i)
    


if __name__=="__main__":
    main()


# plt.imshow(imagenes[0].numpy().squeeze(), cmap='gray_r')