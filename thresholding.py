import cv2

class Thresholding:

    #OTSU 
    def otsu(hist):
        total_pixel = None

        q1 = None
        q2 = None
        u1 = None
        u2 = None

        for i in range(0,total_pixel):
          #q(t) = sum p(i)
          q1 += hist[0][i]

        q2 = total_pixel - q1

        for i in range(0,total_pixel):
          # i * P(i)  
          u1 += (i * hist[0][i]) / q1
          u2 += (i * hist[0][i]) / q2
        """
        for i in range(0,total_pixel):
          Q1 += (hist[0][i]/u1)*(i - q1)^2
          Q2 += (hist[0][i]/u2)*(i - q2)^2
        """

        Q = (q1 * q2) * (u1 - u2)^2
        

        return Q