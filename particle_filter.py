import numpy as np
import math
import random
import cv2
from PIL import Image
import os.path as path
import os

class ParticleFilter:

    def __init__(self, dirpath, M, R, Q, lambda_psi,
        draw_rectangles = True, uWeight = 1):

        self.M = M
        self.R = R
        self.Q = Q
        self.lambda_psi = lambda_psi

        self.draw_rectangles = draw_rectangles

        self.dataset = self.readDataset(dirpath)
        self.imgx = self.dataset[0].shape[1]    # Original image size.
        self.imgy = self.dataset[0].shape[0]
        self.xdim = self.imgx        # Size for scaled images.
        self.ydim = self.imgy

        self.xMeanOld = 0
        self.yMeanOld = 0

        self.uWeight = uWeight


    def scale_images(self, scale):
        self.xdim = int(self.imgx*scale)
        self.ydim = int(self.imgy*scale)
        # old_dataset = self.dataset
        # self.dataset = np.zeros((self.xdim, self.ydim))
        #
        # for i in range(self.dataset.shape[0]):
        #     img = self.dataset[i]
        #     img = cv2.resize(img, (self.xdim, self.ydim))
        #     self.dataset[i] = img


    # reads dataset
    def readDataset(self, dirpath):
        img_list = []
        for name in os.listdir(dirpath):
            if name.endswith('.jpg'):
                img_list.append(name)

        img_list.sort()

        if (path.isdir(dirpath)):
            x = np.array([np.array(Image.open(path.join(dirpath, filename))) for filename in img_list])
            return x
        else:
            return "Error no directory found"


    # pre proccess single image
    def preProc(self, img):
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
                if self.draw_rectangles:
                    cv2.rectangle(
                        img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                pos[0, i] = float(x + w/2)/self.xdim
                pos[1, i] = float(y + h/2)/self.ydim
        else:
            return img, np.array([])

        return img, pos


    def draw_particles(self, S, img):
        for i in range(S.shape[1]):
            cv2.circle(
                img, (int(S[0, i]*self.xdim), int(S[1, i]*self.ydim)), 2, (0, 255, 0), 2)

        xc, yc = self.particle_mean(S)
        cv2.circle(img, (int(xc*self.xdim), int(yc*self.ydim)), 3, (0, 0, 255), 3)

        return img


    def initParticleSet(self, imgWidth, imgHeight):
        x = np.random.uniform(0, imgWidth, self.M)
        y = np.random.uniform(0, imgHeight, self.M)
        w = np.ones(self.M)*(1./self.M)

        S = np.vstack((x, y, w))

        return S


    def particle_mean(self, S):
        return np.mean(S[0, :]), np.mean(S[1, :])


    def getU(self, S, iter):
        #xMean, yMean = self.particle_mean(S)

        if(iter == 3):
            xMean, yMean = self.particle_mean(S)
            self.xMeanOld = xMean
            self.yMeanOld = yMean
        elif iter > 3:
            xMean, yMean = self.particle_mean(S)
            deltaXMean = xMean - self.xMeanOld
            deltaYMean = yMean - self.yMeanOld

            self.yMeanOld = yMean
            self.xMeanOld = xMean

            return self.uWeight * np.array([[deltaXMean], [deltaYMean]])

        return np.array([[0], [0]])



    def predict(self, S, iter):
        """Propagate particles according to motion model. Add diffusion. """
        S_bar = np.zeros(S.shape)

        nx = S.shape[0] - 1

        # Motion model.
        #u = np.array([[0], [0]])
        u = self.getU(S, iter)

        diffusion = np.multiply(np.random.randn(nx, self.M), np.diag(self.R)[np.newaxis, :].T)

        S_bar[0:nx, :] = S[0:nx, :] + u + diffusion
        S_bar[nx, :] = S[nx, :]

        return S_bar


    def systematic_resample(self, S_bar):
        """Perform systematic resampling of the particles. """
        M = S_bar.shape[1]
        nx = S_bar.shape[0] - 1
        S = np.zeros(S_bar.shape)

        density_function = np.cumsum(S_bar[nx, :])   # Cumulative density function.
        r0 = random.random()/M
        r = r0 + (np.arange(1, M + 1) - 1)/float(M) # r0 + (m - 1)/M

        A1 = np.tile(density_function, (M, 1))      # Repeat CDF vertically.
        A2 = np.tile(r[np.newaxis, :].T, (1, M))    # Repeat r horizontally.

        indices = np.argmax(A1 >= A2, axis = 1) # i = min CDF(j) >= r0 + (m - 1)/M

        S = S_bar[:, indices]   # Resample.

        S[nx, :] = 1/float(M)    # Reset weights.

        return S


    def weight(self, S_bar, psi, outlier):
        """Weigh the particles according to the probabilities in psi. """
        psi_inliers = psi[np.invert(outlier), :]    # Discard outlier measurements.

        psi_max = psi[np.argmax(np.sum(psi, 1)), :].reshape(
            (1, S_bar.shape[1]))

        psi_inliers = psi_max

        nx = S_bar.shape[0] - 1
        if psi_inliers.size > 0:

            weights = np.prod(psi_inliers, axis = 0)    # Multiply probabilities.
            weights = weights/np.sum(weights)           # Normalize weights.
            S_bar[nx, :] = weights

        return S_bar


    def associate(self, S_bar, z):
        """Calculate probability of each particle given each measurement. """
        n = z.shape[1]
        nx = S_bar.shape[0] - 1
        dim = nx

        z_pred = np.tile(S_bar[0:dim, :], (1, n)) # [x x ... x]
        z_obs = np.reshape(np.repeat(z, self.M), (dim, n*self.M)) # [z1 ... z1 zn ... zn]

        nu = z_obs - z_pred # True observation minus predicted observation.

        exp_term = -0.5*np.sum(
            np.multiply(np.dot(nu.T, np.linalg.inv(self.Q)).T, nu), axis = 0)
        psis = 1/(2*math.pi*math.sqrt(np.linalg.det(self.Q)))*np.exp(exp_term)

        psi = np.reshape(psis, (n, self.M)) # Rows: measurements, columns: particles.

        outlier = np.mean(psi, axis = 1) < self.lambda_psi

        return outlier, psi


    def run_particle_filter(self):
        S = self.initParticleSet(1, 1)

        for i in range(0, self.dataset.shape[0]):
            print(i)
            img = self.dataset[i]
            img = cv2.resize(img, (self.xdim, self.ydim))#(128, 128))

            if i > 0:
                S_bar = self.predict(S, i)
                img, observation = self.preProc(img)
                if observation.size > 0:

                    outlier, psi = self.associate(S_bar, observation)

                    S_bar = self.weight(S_bar, psi, outlier)

                    S = self.systematic_resample(S_bar)
                else:
                    S = S_bar

                xmean, ymean = self.particle_mean(S)

            img = self.draw_particles(S, img)

            cv2.imshow('', img)
            k = cv2.waitKey(0)
            if k == ord('q'):
                cv2.destroyAllWindows()
                break;


def main():
    path_prefix = '/home/filip/el2320/project/vot2016/'
    path_end = 'iceskater1'

    M = 50
    R_val = 0.02
    Q_val = 0.01
    lambda_psi = 0.1
    scale = 0.75
    uWeight = 0

    draw_rectangles = True

    R = np.diag([1., 1.])*R_val
    Q = np.diag([1., 1.])*Q_val

    directory_path = path_prefix + path_end
    directory_path = 'iceskater1'

    p = ParticleFilter(directory_path, M, R, Q, lambda_psi,
        draw_rectangles = draw_rectangles, uWeight = uWeight)

    p.scale_images(scale)

    p.run_particle_filter()


if __name__ == '__main__':
    main()
