import math
import cv2
import numpy as np
import tensorflow
from skimage import color
from scipy import ndimage
from skimage.measure import label
from scipy import ndimage
import matplotlib
import matplotlib.pyplot as plt
from vektor import distance, point2line


def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)

cnt = -1
def next_id():
    global cnt
    cnt += 1
    return cnt

def get_br_img(broj, frame):

    y1 = broj.y - 7
    y2 = broj.y + 21
    x1 = broj.x - 7
    x2 = broj.x + 21

    br_img = frame[y1:y2, x1:x2]
    br_img_normalizovan = tensorflow.keras.utils.normalize(br_img)

    return br_img_normalizovan

def ne_treba_detektovati(br):
    if (br.x - 16 <= 0):
        return True
    if (br.y - 16 <= 0):
        return True

def brojevi_u_radiusu(broj, brojevi, radius):

    u_radiusu = []
    koo_broj = (int(broj.x), int(broj.y))

    for br in brojevi:

        tracked_item_koo = (int(br.x), int(br.y))
        center_distance = distance(koo_broj, tracked_item_koo)

        if(center_distance < radius):
            u_radiusu.append(br)
    return u_radiusu

def draw_detection_rectangle(br, detected_digit, img):
    y1 = br.y - 17
    y2 = br.y + 11
    x1 = br.x - 16
    x2 = br.x + 12
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.putText(img, str(detected_digit), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    #cv2.imshow('img', img)


def tacka_presla_liniju(point, line):

    line_distance, nearest_point, radius = point2line(point, (line['x1'], line['y1']), (line['x2'], line['y2']))

    return (radius > 0 and line_distance < 9)

def br_presao_liniju(br, line):

    centar = (br.x, br.y)
    bottom_left_corner = (br.x, br.y + br.h // 2)
    return tacka_presla_liniju(centar, line) or tacka_presla_liniju(bottom_left_corner, line)

class Number:

    def __init__(self,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.id = next_id()
        self.predicted_br = None

model = tensorflow.keras.models.load_model('neuronska.model')


def detect_line(img):

    kernel_line = np.ones((2,2),np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    erosion = cv2.erode(gray, kernel_line,iterations = 1)
    #Second and third arguments are our minVal and maxVal respectively.Third argument is aperture_size default 3
    edges = cv2.Canny(erosion,50,150,3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=1, maxLineGap=5)

    x1 = lines[0][0][0]
    y1 = lines[0][0][1]
    x2 = lines[0][0][2]
    y2 = lines[0][0][3]

    return { 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2 }

def get_sum(video_path):

        frame_num = 0;
        video = cv2.VideoCapture(video_path)
        video.set(1, frame_num)
        kernel = np.ones((3, 3), np.uint8)  # strukturni element 2x2 blok

        brojevi = []
        suma = 0

        passed_ids = []
        current_id = 0
        brojac = 0
        suma = 0


        while True:

            frame_num += 1
            ret_val, frame = video.read()

            if not ret_val:
                break

            frame_org = frame.copy()

            lower = np.array([230, 230, 230])
            upper = np.array([255, 255, 255])
            # Threshold the HSV image to get only white colors
            mask = cv2.inRange(frame, lower, upper)

            img_dilate = cv2.dilate(mask, kernel)
            img_dilate = cv2.dilate(img_dilate, kernel)


            image_orig = frame.copy()
            contours, hierarchy = cv2.findContours(img_dilate.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            sorted_regions = []  # lista sortiranih regiona po x osi (sa leva na desno)
            regions_array = []

            line = detect_line(frame)

            x1 = line['x1']
            y1 = line['y1']
            x2 = line['x2']
            y2 = line['y2']

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                area = cv2.contourArea(contour)
                if area > 100 and h < 100 and h > 8 and w > 6:

                    broj = Number(x, y, w, h)

                    if ne_treba_detektovati(broj):
                       continue

                    brojevi_blizu = brojevi_u_radiusu(broj, brojevi, 22)

                    if len(brojevi_blizu) == 0:

                        br_img = get_br_img(broj, mask)
                        predictions = model.predict([[br_img]])
                        predicted_br = np.argmax(predictions[0])
                        broj.predicted_br = predicted_br
                        brojevi.append(broj)
                        continue

                    for broj in brojevi:
                        draw_detection_rectangle(broj, broj.predicted_br, frame)
                        if not br_presao_liniju(broj, line):
                            continue
                        if (passed_ids.__contains__(broj.id)):
                            continue
                        passed_ids.append(broj.id)
                        brojac += 1
                        suma += broj.predicted_br
                        print('brojac:', str(brojac), 'brojac + ' + str(broj.predicted_br), ' = ',
                              str(suma))

                    region = img_dilate[y:y + h + 1, x:x + w + 1]

                    regions_array.append([resize_region(region), (x, y, w, h)])
                    cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

            regions_array = sorted(regions_array, key=lambda item: item[1][0])
            sorted_regions = sorted_regions = [region[0] for region in regions_array]

            cv2.imshow('sel_r', image_orig)

            if cv2.waitKey(1) & 0xFF == ord('c'):
                break
        video.release()
        cv2.destroyAllWindows()
        return suma


def proba():
    prediction_results = []

    for i in range(10):
        video_name = 'video-' + str(i) + '.avi'

        sum = get_sum(video_name)
        prediction_results.append({'video': video_name, 'sum': sum})
        print(video_name, sum)


proba()