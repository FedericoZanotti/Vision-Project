
import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

width_min = 80  # minimum width of rectangle
height_min = 80  # minimum height of rectangle

offset = 6

pos_line = 550

delay = 60  # FPS of vídeo

detected = []
cars = 0


def find_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


car_night = 'yt1s.com - Cars driving at night.mp4'
car_traffic = 'car_traffic_toy.mp4'
cap = cv2.VideoCapture(car_traffic)

bs = cv2.bgsegm.createBackgroundSubtractorMOG()

def draw_and_write(x, y, w, h, img, temp_l=[], id=-1, first_frame=False):
    font = cv2.FONT_HERSHEY_PLAIN
    fontscale = 2
    thickness = 2


    if first_frame:  # first frame or object is not longer in the video
        t_size = cv2.getTextSize(str(id), font, fontscale, 1)[0]
        newXplusW= x + t_size[0] + 3
        newYplusH = y + t_size[1] + 4
        cv2.rectangle(img,  (x, y),  (newXplusW, newYplusH), (0, 255, 0), -1)
        cv2.putText(img, str(id), (x,y + t_size[1] + 4), font,
                    fontscale, [225, 255, 255], thickness)
    else:
        t_size = cv2.getTextSize(str(id), font, fontscale, 1)[0]
        newXplusW = x + t_size[0] + 3
        newYplusH = y + t_size[1] + 4
        cv2.rectangle(img,  (x, y),  (newXplusW, newYplusH), (0, 255, 0), -1)
        cv2.putText(img, str(temp_l[0]), (x, y + t_size[1] + 4), font,
                    fontscale, [225, 255, 255], thickness)

def start_process():
    global cars
    count =0
    centers={}
    num_frame=0
    passed=[]
    while True:
        ret, frame1 = cap.read()
        tempo = float(1 / delay)
        sleep(tempo)
        grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (3, 3), 5)
        img_sub = bs.apply(blur)
        d = cv2.dilate(img_sub, np.ones((4, 4)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        dilatated = cv2.morphologyEx(d, cv2.MORPH_RECT, kernel)
        dilatated = cv2.morphologyEx(dilatated, cv2.MORPH_RECT, kernel)
        contour, _ = cv2.findContours(dilatated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.line(frame1, (0, pos_line), (frame1.shape[1], pos_line), (255, 127, 0), 3)

        for (i, c) in enumerate(contour):
            (x, y, w, h) = cv2.boundingRect(c)
            valid_contour = (w >= width_min) and (h >= height_min)
            if not valid_contour:
                continue

            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center = find_center(x, y, w, h)
            updated = False
            temp_l = [12, (1, 1)]
            if not bool(centers):
                count += 1
                centers[count] = center
                updated=True
                draw_and_write(x, y, w, h,frame1, id=count, first_frame=True)

            else:
                best_d = 100
                for k, v in centers.items():  # for every car compute euclidean distance between centers of new object and the objects in memory
                    x = v[0]
                    y = v[1]
                    x_new = center[0]  # new object to update
                    y_new = center[1]
                    euclidian_distance = np.sqrt(np.power(y - y_new, 2) + np.power(x - x_new, 2))
                    if euclidian_distance < best_d:  # if statement to find minimum euclidean distance
                        best_d = euclidian_distance
                        temp_l[0] = k  # temp list = [id_obj, (x,y)] id_obj is the one with best euclidean distance.
                        temp_l[1] = (x_new, y_new)
                if best_d < 40:  # if the best value as an important value (<40 for car_traffic_toy ) (<20 for car night)
                    centers[temp_l[0]] = temp_l[1]  # update object
                    updated = True
                    draw_and_write(x, y, w, h,frame1, id=count, temp_l=temp_l)

            if not updated and num_frame != 0:  # if not updated we have another object
                count += 1
                centers[count] = centro
                draw_and_write(x,y,w, h, frame1, id = count, first_frame=True)


            y_test = int(temp_l[1][1])

            if y_test < (pos_line + offset) and y_test > (pos_line - offset):
                id_vehicle = list(centers.keys())[list(centers.values()).index(temp_l[1])]
                print(f"I have seen vehicle {id_vehicle} passing trough the line")
                if len(passed) == 0:
                    passed.append(id_vehicle)
                elif id_vehicle not in passed:
                    passed.append(id_vehicle)
                else:
                    pass
        draw_x = int(frame1.shape[0] * 0.05)
        draw_y = int(frame1.shape[1] * 0.05)
        cv2.putText(frame1, 'vehicle counted: ' + str(count), (draw_x, draw_y), cv2.FONT_HERSHEY_SIMPLEX,
                        1, [225, 225, 225], 2)
        cv2.imshow("Original Video", frame1)


        if cv2.waitKey(33) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start_process()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
