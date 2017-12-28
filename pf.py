import cv2
import numpy as np
from PIL import Image
import os.path as path
import os
import particle


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
    else:
        return img, np.array([])

    return img, pos

def draw_particles(S, img):
    for i in range(S.shape[1]):
        cv2.circle(img, (int(S[0, i]), int(S[1, i])), 2, (0, 256, 0), 2)

    return img


def initParticleSet(imgWidth, imgHeight, M):
    x = np.random.uniform(0, imgWidth, M)
    y = np.random.uniform(0, imgHeight, M)
    w = np.ones(M)*(1./M)
    #print(x,y, w)
    S = np.vstack((x, y, w))

    return S


def particle_mean(S):
    return np.mean(S[0, :]), np.mean(S[1, :])

if __name__ == '__main__':
    directorypath = '/home/filip/el2320/project/vot2016/graduate'

    # read dataset
    dataset = readDataset(directorypath)

    # init particle set
    M = 50
    R_val = 10
    xdim = dataset[0].shape[1]
    ydim = dataset[0].shape[0]
    S = initParticleSet(xdim, ydim, M)
    img = dataset[0]
    img = draw_particles(S, img)
    cv2.imshow('', img)
    k = cv2.waitKey(0)
    if k == ord('q'):
        cv2.destroyAllWindows()

    R = np.diag([1., 1.])
    Q = np.diag([100, 100])*0.1


    for i in range(dataset.shape[0]):
        print(i),
        img = dataset[i]
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #img = cv2.resize(img, (128, 128))
        img, observation = preProc(img)

        S_bar = particle.predict(S, 0, R, 0)

        if observation.size > 0:

            outlier, psi = particle.associate(S_bar, observation, 0.00001, Q)
            print(outlier)

            S_bar = particle.weight(S_bar, psi, outlier)

            S = particle.systematic_resample(S_bar)
        else:
            S = S_bar
            print()

        xmean, ymean = particle_mean(S)
        #print('x: {}, y: {}'.format(xmean, ymean))

        img = draw_particles(S, img)

        #particle stuff



        cv2.imshow('', img)
        k = cv2.waitKey(0)
        if k == ord('q'):
            cv2.destroyAllWindows()
            break;
