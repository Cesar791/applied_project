import cv2
import numpy as np
from PIL import Image
import os.path as path
import os
from particle1 import predict


# reads dataset
def readDataset(dirpath):
    img_list = []
    for name in os.listdir(dirpath):
        if name.endswith('.jpg'):
            img_list.append(name)

    img_list.sort()

    if (path.isdir(directorypath)):
        x = np.array([np.array(Image.open(path.join(dirpath, filename))) for filename in img_list])
        return x
    else:
        return "Error no directory found"


# pre proccess single image
def preProc(img):
    # init detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # detect people
    (rects, weights) = hog.detectMultiScale(img, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)


    # draw bounding boxes
    if np.size(rects) > 0:
        # row position, column new rect
        pos = np.zeros((2, rects.shape[0]))

        for i, (x, y, w, h) in enumerate(rects):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            pos[0, i] = x + w/2
            pos[1, i] = y + h/2
            print('x: {}, y: {}'.format(x + w/2, y + h/2))
    else:
        return img, np.array([])

    return img, pos


def initParticleSet(imgWidth, imgHeight, M):
    x = np.random.uniform(0, imgWidth, M)
    y = np.random.uniform(0, imgHeight, M)
    w = np.ones(M)*(1./M)
    #print(x,y, w)
    S = np.array([x, y, w])
    print(S)
    return S

if __name__ == '__main__':
    directorypath = 'gymnastics1'

    # read dataset
    dataset = readDataset(directorypath)

    # init particle set
    S_bar = initParticleSet(128, 128, 50)


    for i in range(dataset.shape[0]):
        img = dataset[i]
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #img = cv2.resize(img, (128, 128))
        #img, observation = preProc(img)

        #particle stuff



        cv2.imshow('', img)
        k = cv2.waitKey(25)
        if k == ord('q'):
            cv2.destroyAllWindows()
            break;
