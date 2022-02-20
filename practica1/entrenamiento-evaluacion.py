import AMPEdnn as dnn
import os.path
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

path = '/home/javi/Documentos/UNI/AMPE'
WEIGHTS_PATH = './my_weights.pt'

def main():
    
    # obtener el dataset de minst
     
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),])
    trainset = datasets.MNIST(path, download=True, train=True, transform=transform)
    valset = datasets.MNIST(path, download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # crear la dnn a partir del model del archivo AMPEdnn.py   
    model = dnn.My_DNN()
    if (os.path.isfile(WEIGHTS_PATH)):
        
        print('Se ha detectado un fichero de pesos. Se ha cargado al modelo.')
        model.load_state_dict(torch.load(WEIGHTS_PATH))
        # se infieren 3 imágenes a la dnn entrenada para probar la eficacia.
        for i in range(3):
            images, labels = next(iter(valloader))
            img = images[0].view(1, 784)
            with torch.no_grad():
                logps = model(img)

            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            print("\nPredicted Digit =", probab.index(max(probab)))
            print("True Digit =", labels.numpy()[0])
            # plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
    
    else:
        
        print('No se ha detectado ningún fichero de pesos "my_weights.pt" en el directorio. Comenzando el entrenamiento de la DNN...')
        images, labels = next(iter(valloader))
        criterion = nn.NLLLoss()
        images = images.view(images.shape[0], -1)
        logps = model(images) #log probabilities
        loss = criterion(logps, labels) #calculate the NLL loss
        loss.backward()
        
        optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
        time0 = time()
        epochs = 25
        for e in range(epochs):
            running_loss = 0
            for images, labels in trainloader:
                # Flatten MNIST images into a 784 long vector
                images = images.view(images.shape[0], -1)

                # Training pass
                optimizer.zero_grad()

                output = model(images)
                loss = criterion(output, labels)

                #This is where the model learns by backpropagating
                loss.backward()

                #And optimizes its weights here
                optimizer.step()
                running_loss += loss.item()
            else:
                print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
        print("\nTiempo transcurrido en entrenar la red (en minutos) =",(time()-time0)/60)
        
        images, labels = next(iter(valloader))
        img = images[0].view(1, 784)
        with torch.no_grad():
            logps = model(img)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        print("Predicted Digit =", probab.index(max(probab)))
        
        correct_count, all_count = 0, 0
        for images,labels in valloader:
            for i in range(len(labels)):
                img = images[i].view(1, 784)
                with torch.no_grad():
                    logps = model(img)

                ps = torch.exp(logps)
                probab = list(ps.numpy()[0])
                pred_label = probab.index(max(probab))
                true_label = labels.numpy()[i]
                if(true_label == pred_label):
                    correct_count += 1
                all_count += 1

        print("Número total de imágenes testeadas =", all_count)
        print("\nPrecisión de la DNN sobre el conjunto de validación: ", (correct_count/all_count))

        print('Guardando pesos en ',WEIGHTS_PATH)
        torch.save(model.state_dict(), WEIGHTS_PATH)

if __name__=="__main__":
    main()