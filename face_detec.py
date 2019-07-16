import cv2 as cv
import numpy as np
import os
from PIL import Image


path = "/usr/local/lib/python3.6/dist-packages/cv2/data/"


def face_detect(filepath):
    image = cv.imread(filepath)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier(path+"haarcascade_frontalface_default.xml")
    faces = face_detector.detectMultiScale(gray, 1.1, 2)
    result = []
    for x, y, w, h in faces:
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        result.append((x, y, x+w, y+h))
    cv.imshow("result", image)
    return result


def face_save(image, root):
    faces = face_detect(image)
    if faces:
        save_name = image.split('.')[0] + "_faces"
        print(save_name.split('/'))
        # exit()
        for (x1, y1, x2, y2) in faces:
            file_name = save_name+'.jpg'
            Image.open(image).crop((x1, y1, x2, y2)).resize((100, 100)).save(file_name)


def turn_gray(filepath):
    Image.open(filepath).convert('L').save(filepath)


if __name__ == '__main__':
    # filepath = "/home/victor/图片/新建文件夹 (2)"
    filepath = "/home/victor/图片/新建文件夹 (2)/data"
    for root, dirs, files in os.walk(filepath):
        for dir in dirs:
            print(dir)
            i = 0
            for root_2, dirs_2, files_2 in os.walk(root+'/'+dir):
                for file in files_2:
                    #######################################################
                    # new_name = filepath+'/'+dir+'/'+dir+'-'+str(i)+'.jpg'
                    # print(new_name)
                    # os.rename(root_2+'/'+file, new_name)
                    #######################################################
                    face_save(root_2+'/'+file, root_2)
                    #######################################################
                    # turn_gray(root_2+'/'+file)
                    i += 1


# cv.waitKey(0)
#
# cv.destroyAllWindows()
