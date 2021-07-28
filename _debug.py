from load_cifar import load_cifar
from load_lfw import load_lfw
from PIL import Image
import cv2

x, xt, y, yt = load_lfw(normalize=False)
#print(X_train)
#print(X_train.shape)

img = x[355].transpose(1,2,0)
#print(img)
print(img.shape)
import matplotlib.pyplot as plt
#plt.imshow(img.transpose(1,2,0))
#plt.show()

grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Grayscale', grayscale)
#cv2.waitKey()
plt.imsave("imgs/resized/355.png", grayscale, cmap="gray")
#plt.imshow(grayscale, cmap="gray")
#plt.show()

x, xt, y, yt = load_lfw(normalize=True)

img = x[355].transpose(1,2,0)
#print(img)
print(img.shape)
import matplotlib.pyplot as plt
#plt.imshow(img.transpose(1,2,0))
#plt.show()

grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Grayscale', grayscale)
#cv2.waitKey()
plt.imsave("imgs/resized/355_norm.png", grayscale, cmap="gray")
