
import cv2
import numpy as np
from time import sleep


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

def find_center(x, y, w, h):
        x1 = int(w / 2)
        y1 = int(h / 2)
        cx = x + x1
        cy = y + y1
        return cx, cy


width_min = 80  # minimum width of rectangle
height_min = 80  # max height of rectangle

offset = 6

pos_line = 350

delay = 60  # FPS of video

detected = []
cars = 0


car_night = 'car_traffic_night_2_Trim.mp4'
car_traffic = 'car_traffic_toy.mp4'
cap = cv2.VideoCapture(car_night)

bs = cv2.bgsegm.createBackgroundSubtractorMOG()  


def start_process():

    global cars
    num_frame=0
    codec = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('./nigh_try.avi', codec, fps, (width, height))
    while True:
        ret, frame1 = cap.read()
        num_frame+=1

        tempo = float(1 / delay)
        sleep(tempo)
        grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (3, 3), 5)
        img_sub = bs.apply(blur)
        d = cv2.dilate(img_sub, np.ones((4,4)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
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
            detected.append(center)

            for (x, y) in detected:
                if y < (pos_line + offset) and y > (pos_line - offset):
                    cars += 1
                    cv2.line(frame1, (25, pos_line), (1200, pos_line), (0, 127, 255), 3)
                    detected.remove((x, y))
                    print("car is detected : " + str(cars))

        cv2.putText(frame1, "VEHICLE COUNT : " + str(cars), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        cv2.imshow("Original Video", frame1)
        cv2.imshow("BS", dilatated)
        print(num_frame)
        out.write(frame1)

        if cv2.waitKey(1) == 27:
            break
        if num_frame==2500:
            break


    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    start_process()

