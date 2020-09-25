# # import cv2

# # def funcBrightContrast(bright=0):
# #     bright = cv2.getTrackbarPos('bright', 'Life2Coding')
# #     contrast = cv2.getTrackbarPos('contrast', 'Life2Coding')

# #     effect = apply_brightness_contrast(img,bright,contrast)
# #     cv2.imshow('Effect', effect)

# # def apply_brightness_contrast(input_img, brightness = 255, contrast = 127):
# #     brightness = map(brightness, 0, 510, -255, 255)
# #     contrast = map(contrast, 0, 254, -127, 127)

# #     if brightness != 0:
# #         if brightness > 0:
# #             shadow = brightness
# #             highlight = 255
# #         else:
# #             shadow = 0
# #             highlight = 255 + brightness
# #         alpha_b = (highlight - shadow)/255
# #         gamma_b = shadow

# #         buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
# #     else:
# #         buf = input_img.copy()

# #     if contrast != 0:
# #         f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
# #         alpha_c = f
# #         gamma_c = 127*(1-f)

# #         buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

# #     cv2.putText(buf,'B:{},C:{}'.format(brightness,contrast),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
# #     return buf

# # def map(x, in_min, in_max, out_min, out_max):
# #     return int((x-in_min) * (out_max-out_min) / (in_max-in_min) + out_min)

# # if __name__ == '__main__':

# #     original = cv2.imread("Einstein.bmp", 1)
# #     img = original.copy()

# #     cv2.namedWindow('Life2Coding',1)

# #     bright = 255
# #     contrast = 127

# #     #Brightness value range -255 to 255
# #     #Contrast value range -127 to 127

# #     cv2.createTrackbar('bright', 'Life2Coding', bright, 2*255, funcBrightContrast)
# #     cv2.createTrackbar('contrast', 'Life2Coding', contrast, 2*127, funcBrightContrast)
# #     funcBrightContrast(0)
# #     cv2.imshow('Life2Coding', original)


# # cv2.waitKey(0)
# import cv2

# #-----Reading the image-----------------------------------------------------
# img = cv2.imread('city.jpg', 1)
# # cv2.imshow("img",img) 

# #-----Converting image to LAB Color model----------------------------------- 
# lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# # cv2.imshow("lab",lab)

# #-----Splitting the LAB image to different channels-------------------------
# l, a, b = cv2.split(lab)
# # cv2.imshow('l_channel', l)
# # cv2.imshow('a_channel', a)
# # cv2.imshow('b_channel', b)

# #-----Applying CLAHE to L-channel-------------------------------------------
# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
# cl = clahe.apply(l)
# # cv2.imshow('CLAHE output', cl)

# #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
# limg = cv2.merge((cl,a,b))
# # cv2.imshow('limg', limg)

# #-----Converting image from LAB Color model to RGB model--------------------
# final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
# cv2.imshow('final', final)

# # inputmn = input()
import numpy as np
from skimage import io
from cv2 import cv2
image = io.imread('im2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# io.imshow(gray)
io.imsave('im2_before.jpg',gray)
def getLine (y1,y2,x1,x2):
    y=  y2-y1
    x=  x2-x1
    slope = (y/x)  
    c = y1-slope*x1
    return (slope,c)

def IncreaseContrast(A,B,C,D, OriginalIm):
   
    NewIm = cv2.equalizeHist(OriginalIm)

                
    return NewIm

Image1 = IncreaseContrast(30.0,20.0,180.0,230.0,gray)
# Image1 = IncreaseContrast(70.0,20.0,140.0,240.0,gray)

# io.imshow(Image1)
io.imsave('im2_increased.jpg',Image1)