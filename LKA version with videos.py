
import cv2
import numpy as np
import time

def display_text_on_image(image, param):

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 255)
    text = text = "L'erreur est : " + str(param)
    text_size, _ = cv2.getTextSize(text, font, 1, 2)
    cv2.putText(image, text, (570, 250), font, 1, color, 2)

def car_shape(image):
    height, width, _ = image.shape
    pts = np.array([[482, 718], [557, 638], [935, 638], [1008, 718]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(image, [pts], True, (255, 200, 2001), 2)
    car_center = (482+1008)//2
    cv2.line(image, (car_center, 718), (car_center, int(height * 0.7)), (112, 200, 1), 3)#////la ligne du milieu de la voiture
    return frame,car_center

def draw_middle_line(image, lines, smooth_factor=25):
    height, width, _ = image.shape
    # print(height,"",width)
    left_line, right_line = lines
    left_x_bottom, _, left_x_top, _ = left_line
    right_x_bottom, _, right_x_top, _ = right_line
    mid_x_bottom = (left_x_bottom + right_x_bottom) // 2
    print("la largeur de la voie est ",right_x_bottom - left_x_bottom )
    mid_y_bottom = height
    mid_x_top = (left_x_top + right_x_top) // 2
    mid_y_top = int(height * 0.7)

    if not hasattr(draw_middle_line, "prev_mid_x_bottom"):
        draw_middle_line.prev_mid_x_bottom = []
    if not hasattr(draw_middle_line, "prev_mid_x_top"):
        draw_middle_line.prev_mid_x_top = []

    draw_middle_line.prev_mid_x_bottom.append(mid_x_bottom)
    draw_middle_line.prev_mid_x_top.append(mid_x_top)

    if len(draw_middle_line.prev_mid_x_bottom) > smooth_factor:
        draw_middle_line.prev_mid_x_bottom.pop(0)
        draw_middle_line.prev_mid_x_top.pop(0)
    smoothed_mid_x_bottom = int(sum(draw_middle_line.prev_mid_x_bottom) / len(draw_middle_line.prev_mid_x_bottom))
    smoothed_mid_x_top = int(sum(draw_middle_line.prev_mid_x_top) / len(draw_middle_line.prev_mid_x_top))

    cv2.line(image, (int(mid_x_bottom), int(mid_y_bottom)), (int(mid_x_top), int(mid_y_top)), (255, 0, 0), 2)
    cv2.line(image, (int(mid_x_top), int(718)), (int(mid_x_top), int(mid_y_top)), (255, 111, 0), 2)
    return mid_x_top
def make_coordinates(image,line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3.5/5))
    if not slope:
        slope=0.01
    else:
     x1 = int((y1-intercept)/slope)
     x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image,lines):
    left_fit = []#liste vide
    right_fit = []#liste vide

    for line in lines :
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit,axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    if not left_fit:
        left_line = [0, 0, 0, 0]
    else:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
    if not right_fit:
        right_line = [0, 0, 0, 0]
    else:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)

    return np.array([left_line,right_line])

def Process1(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 165, 220, cv2.THRESH_BINARY)
    canny = cv2.Canny(thresh, 150 , 200)
    return canny
def display_lines(image,lines,erreur, left_line, right_line):
    line_image = np.zeros_like(image)
    if lines is not None:
            if erreur < -20:
             for lin2 in left_line:
                    x1, y1, x2, y2 = left_line.reshape(4)
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 250), 10)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    color = (0, 0, 255)
                    text = text = "Attention depart hors de la voie"
                    text_size, _ = cv2.getTextSize(text, font, 1, 2)
                    cv2.putText(image, text, (500, 100), font, 1, color, 2)
            elif erreur > 20:
                 for lin1 in right_line:
                     x1, y1, x2, y2 = right_line.reshape(4)
                     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                     cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 250), 10)
                     font = cv2.FONT_HERSHEY_SIMPLEX
                     color = (0, 0, 255)
                     text = text = "Attention depart hors de la voie"
                     text_size, _ = cv2.getTextSize(text, font, 1, 2)
                     cv2.putText(image, text, (500, 100), font, 1, color, 2)

            else:
              for line in lines:
                 x1, y1, x2, y2 = line.reshape(4)
                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                 cv2.line(line_image, (x1, y1), (x2, y2), (0, 250, 0), 10)



    return line_image



def region_of_interest(image):
    tl=(242,622)
    bl=(616,444)
    tr=(833,435)
    br=(1090,600)
    polygons  = np.array([[tl,bl,tr,br] ])

    cv2.circle(frame,tl,5,(255,255,0),-1)
    cv2.circle(frame,bl,5,(255,255,0),-1)
    cv2.circle(frame,tr,5,(255,255,0),-1)
    cv2.circle(frame,br,5,(255,255,0),-1)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, (255,255,255))
    masked_image= cv2.bitwise_and(image,mask)
    return masked_image
alpha = 0.2
prev_lines = None
erreur = 0
#acquisition de la vidéo
cap = cv2.VideoCapture("test17.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = Process1(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=70,maxLineGap=30)
    if prev_lines is not None:
        smoothed_lines = alpha * average_slope_intercept(frame, lines) + (1 - alpha) * prev_lines
    else:
        smoothed_lines = average_slope_intercept(frame, lines)
    prev_lines = smoothed_lines
    left_line,right_line = smoothed_lines
    line_image = display_lines(frame, smoothed_lines, erreur, left_line, right_line)
    draw_middle_line(line_image, smoothed_lines)
    car_shapeframe= car_shape(frame)
    coordmidline = draw_middle_line(line_image, smoothed_lines)
    erreur = car_shapeframe[1] - draw_middle_line(line_image, smoothed_lines)
    display_lines(frame, smoothed_lines, erreur, left_line, right_line)
    cv2.line(frame, (car_shapeframe[1], 570), (int(coordmidline), 570), (0, 0, 255), 5)
    display_text_on_image(frame,erreur)
    combo_image = cv2.addWeighted(car_shapeframe[0], 0.8, line_image, 1, 1)
    cv2.imshow("result",combo_image)  # <--
    if cv2.waitKey(1)==ord('c'):
        break
cap.release()
cv2.destroyAllWindows()



