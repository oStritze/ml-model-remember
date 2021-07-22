import numpy as np

from _lfw import *


def _load_image_names():
    """
    Manually created list of images.
    """
    f = open("lfw_home/list_of_images.txt")
    images = f.readlines()
    return images

def _load_female_names():
    """
    Load female image names from manually curated list.
    from http://vis-www.cs.umass.edu/lfw/#resources 
    """
    f = open("lfw_home/female_names.txt")
    females = f.readlines()
    return females

def _create_gender_target(imagenames, females):
    """
    """

    gender = np.array([])
    for img in imagenames:
        if img in females:
            gender = np.append(gender, 1)
        else:
            gender = np.append(gender, 0)
    return gender

def load_lfw(min_faces=0):
    #from sklearn.datasets import fetch_lfw_people
    lfw_people = fetch_lfw_people(min_faces_per_person=min_faces, resize=0.5, data_home="./", 
                                download_if_missing=True)

    images = _load_image_names()
    females = _load_female_names()
    target = _create_gender_target(images, females) # create gender target

    assert(target.shape == lfw_people.target.shape)
    
    return lfw_people.data, target

if __name__ == '__main__':
    #pass
    
    X,y = load_lfw()
    print(y)
    print(y.sum())
    print(y.shape-y.sum())

    from PIL import Image
    i = 0
    print(X[i])
    pil_image=Image.fromarray(X[i].reshape(62,47))
    pil_image.show()

