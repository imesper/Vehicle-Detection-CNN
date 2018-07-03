
from keras.models import load_model
import h5py
from keras import __version__ as keras_version
import numpy as np
from helpers import slide_window, draw_boxes
from improvements import add_heat, apply_threshold, draw_labeled_bboxes
from scipy.ndimage.measurements import label
import cv2
import sobel
import color_masks
from line import Line
from lane_detection import LaneDetection
from moviepy.editor import VideoFileClip
from camera_calibration import camera_calibration, undistort_image, test_calibration


def get_avarage_xfitted(fitsx, smooth=5):
    average = 0
    if len(fitsx) == smooth:
        for fitx in fitsx:
            average += fitx
        average /= smooth
    return average


def predict(model, image):
    # check that model Keras version is same as local Keras version

    image_array = np.asarray(image)

    return model.predict(image_array[None, :, :, :], batch_size=1)


rightLine = [0] * 4
leftLine = [0] * 4
count = 0
smooth = 5

mtx, dist = camera_calibration()

leftLines = []
rightLines = []
lastLeftLine = Line()
lastRightLine = Line()

cap = cv2.VideoCapture('./project_video.mp4')

detection = LaneDetection(nwindows=20)

fourcc = cv2.VideoWriter_fourcc('X', '2', '6', '4')
out = cv2.VideoWriter('output_videos/output_video.mp4', fourcc,
                      cap.get(cv2.CAP_PROP_FPS), (1280, 720))

model = load_model('./models/model-008.h5')

previous_windows = None

while(True):
    # Capture frame-by-frame
    ret, image = cap.read()

    if ret == True:

        detection.setImage(image)

        left_fit, right_fit, left_curvature, right_curvature, left_fitx, right_fitx, dif_meters, left_base, right_base = detection.process_detection(
            mtx, dist)
        sanity = True
        if lastLeftLine.detected and lastRightLine.detected:

            lastLeftLine.diffs = lastLeftLine.current_fit - left_fit
            lastLeftLine.line_base_pos = left_base
            lastLeftLine.current_fit = left_fit
            lastLeftLine.radius_of_curvature = left_curvature
            lastLeftLine.recent_xfitted.append(left_fitx)
            if len(lastLeftLine.recent_xfitted) > smooth:
                lastLeftLine.recent_xfitted.pop(0)
            if len(lastLeftLine.recent_xfitted) == smooth:
                lastLeftLine.bestx = get_avarage_xfitted(
                    lastLeftLine.recent_xfitted)
            lastRightLine.diffs = lastRightLine.current_fit - right_fit
            lastRightLine.line_base_pos = right_base
            lastRightLine.current_fit = right_fit
            lastRightLine.radius_of_curvature = right_curvature
            lastRightLine.recent_xfitted.append(right_fitx)
            if len(lastRightLine.recent_xfitted) > smooth:
                lastRightLine.recent_xfitted.pop(0)
            if len(lastRightLine.recent_xfitted) == smooth:
                lastRightLine.bestx = get_avarage_xfitted(
                    lastRightLine.recent_xfitted)
            image = detection.draw_final_image(lastLeftLine, lastRightLine)
        else:
            lastLeftLine.detected = True
            lastLeftLine.line_base_pos = left_base
            lastLeftLine.current_fit = left_fit
            lastLeftLine.best_fit = left_fit
            lastLeftLine.bestx = left_fitx
            lastLeftLine.radius_of_curvature = left_curvature
            lastLeftLine.recent_xfitted.append(left_fitx)

            lastRightLine.detected = True
            lastRightLine.line_base_pos = right_base
            lastRightLine.current_fit = right_fit
            lastRightLine.best_fit = right_fit
            lastRightLine.bestx = right_fitx

            lastRightLine.radius_of_curvature = right_curvature
            lastRightLine.recent_xfitted.append(right_fitx)
            image = detection.draw_final_image(lastLeftLine, lastRightLine)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        y_start_stop = (380, 665)
        windows = []
        windows.extend(slide_window(image, x_start_stop=[None, None], y_start_stop=(400, 665),
                                    xy_window=(64, 64), xy_overlap=(0.75, 0.75)))
        windows.extend(slide_window(image, x_start_stop=[None, None], y_start_stop=(400, 464),
                                    xy_window=(32, 32), xy_overlap=(0.5, 0.5)))
        windows.extend(slide_window(image, x_start_stop=[None, None], y_start_stop=(500, 665),
                                    xy_window=(165, 165), xy_overlap=(0.5, 0)))
        on_windows = []

        for window in windows:

            # 3) Extract the test window from original image
            test_img = cv2.resize(
                image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64), interpolation=cv2.INTER_CUBIC)
            prediction = predict(model, test_img)

            if prediction > 0.98:
                print(prediction)
                on_windows.append(window)

        tmp = on_windows.copy()
        # print(len(on_windows), len(tmp))
        if previous_windows != None:
            on_windows.extend(previous_windows)
        previous_windows = tmp

        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        # np.set_printoptions(threshold=np.nan)
        # print(box_windows)
        # Add heat to each box in box list
        heat = add_heat(heat, on_windows)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 1)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)

        box_img = draw_labeled_bboxes(np.copy(image), labels)
        final_image = cv2.cvtColor(box_img, cv2.COLOR_RGB2BGR)

        # final_image = final_image.astype(np.float32)*255
        name_frame = 'final_'+str(count) + '.png'

        out.write(final_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
