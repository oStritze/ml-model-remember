import numpy as np
import math

from _lfw import * # sklearn lfw code but without shuffling loaded images
from sklearn.model_selection import train_test_split

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

def _normalize(X_train, X_test):
    # normalize mean
    pixel_mean = np.mean(X_train, axis=0)
    X_train -= pixel_mean
    X_test -= pixel_mean
    return X_train, X_test

def load_lfw(min_faces=0, shuffle=True, resize=0.5):
    #from sklearn.datasets import fetch_lfw_people
    lfw_people = fetch_lfw_people(min_faces_per_person=min_faces, resize=resize, data_home="./", 
                                download_if_missing=True)

    images = _load_image_names()
    females = _load_female_names()
    target = _create_gender_target(images, females) # create gender target

    assert(target.shape == lfw_people.target.shape)
    
    data = lfw_people.data
    data = data/255 # scale to [0,1]

    N = data.shape[0]
    H = math.ceil(124 * resize)
    W = math.ceil(94 * resize)
    data = data.reshape(N, H, W)

    if shuffle:
        ind = np.arange(target.shape[0])
        np.random.RandomState(42).shuffle(ind)
        data = data[ind]
        target = target[ind]

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=42, stratify=target)

    X_train, X_test = _normalize(X_train, X_test)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    pass
    
    """
    # some debug stuff
    x, xt, y, yt = load_lfw()
    print(y)
    print(y.sum())
    print(y.shape-y.sum())

    from PIL import Image
    i = -3
    print(x[i]*255)
    pil_image=Image.fromarray(x[i]*255)
    pil_image.show()
    """