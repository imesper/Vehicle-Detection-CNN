import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import os
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.externals import joblib
from helpers import *
from improvements import add_heat, apply_threshold, draw_labeled_bboxes
from scipy.ndimage.measurements import label
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images


def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())


def get_hog_from_windown(window, hog_features, hog_channel=0):
    print(window, len(hog_features[0][window[0][1]: window[1]
                                      [1], window[0][0]:window[1][0]].ravel()))
    if hog_channel == "ALL":
        hog_feat1 = hog_features[0][window[0][1]:window[1]
                                    [1], window[0][0]:window[1][0]].ravel()
        hog_feat2 = hog_features[1][window[0][1]:window[1]
                                    [1], window[0][0]:window[1][0]].ravel()
        hog_feat3 = hog_features[2][window[0][1]:window[1]
                                    [1], window[0][0]:window[1][0]].ravel()
        print(hog_feat1)
        return np.hstack((hog_feat1, hog_feat2, hog_feat3))
    else:
        return hog_features[hog_channel][window[0][1]:window[1][1], window[0][0]:window[1][0]].ravel()


def search_windows(img, windows, clf, scaler, hog_features, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):

    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        img_features = []
        # 3) Extract the test window from original image
        test_img = cv2.resize(
            img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64), interpolation=cv2.INTER_CUBIC)
        # 4) Extract features for that window using single_img_features()
        if hog_feat == True:
            hog_features = get_hog_from_windown(
                window, hog_features, hog_channel)
            print(hog_features)
            # 4) Append features to list
            # img_features.append(hog_features)
        # 3) Compute spatial features if flag is set
        if spatial_feat == True:
            spatial_features = bin_spatial(test_img, size=spatial_size)
            # 4) Append features to list
            # img_features.append(spatial_features)
        # 5) Compute histogram features if flag is set
        if hist_feat == True:
            hist_features = color_hist(test_img, nbins=hist_bins)
            # 6) Append features to list
            # img_features.append(hist_features)
        # features = single_img_features(test_img, color_space=color_space,
        #                                spatial_size=spatial_size, hist_bins=hist_bins,
        #                                orient=orient, pix_per_cell=pix_per_cell,
        #                                cell_per_block=cell_per_block,
        #                                hog_channel=hog_channel, spatial_feat=spatial_feat,
        #                                hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        # print(np.array(features).reshape(1, -1).shape)
        # inds = np.where(np.isnan(img_features))

        # # Place column means in the indices. Align the arrays using take
        # img_features[inds] = 0
        h_features = np.hstack(
            (spatial_features, hist_features, hog_features)).reshape(1, -1)
        inds = np.where(np.isnan(h_features))
        # print(inds)
        # Place column means in the indices. Align the arrays using take
        h_features[inds] = 0
        test_features = X_scaler.transform(h_features.reshape(1, -1))
        # test_features = scaler.transform(np.array(h_features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


def train_SVM(save_file=True, file_path='./models/model.plk', scaler_file_path='./models/scaler.plk', color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):

    # Read in cars and notcars
    cars = []
    notcars = []

    for file in os.listdir('./vehicles'):
        if os.path.isdir('./vehicles/'+file):
            for filename in os.listdir('./vehicles/'+file):
                if '.png' in filename:
                    cars.append('./vehicles/'+file+'/'+filename)

    for file in os.listdir('./non-vehicles'):
        if os.path.isdir('./non-vehicles/'+file):
            for filename in os.listdir('./non-vehicles/'+file):
                if '.png' in filename:
                    notcars.append('./non-vehicles/'+file+'/'+filename)

    car_features, image_hog_car = extract_features(cars, color_space=color_space,
                                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                                   orient=orient, pix_per_cell=pix_per_cell,
                                                   cell_per_block=cell_per_block,
                                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                                   hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features, image_hog_noncar = extract_features(notcars[200:206], color_space=color_space,
                                                         spatial_size=spatial_size, hist_bins=hist_bins,
                                                         orient=orient, pix_per_cell=pix_per_cell,
                                                         cell_per_block=cell_per_block,
                                                         hog_channel=hog_channel, spatial_feat=spatial_feat,
                                                         hist_feat=hist_feat, hog_feat=hog_feat)

    # for i, car in enumerate(image_hog_car):
    #     cv2.imwrite('./output_images/hog_car_' + str(i) + '.png', car)
    # for i, noncar in enumerate(image_hog_noncar):
    #     cv2.imwrite('./output_images/hog_noncar_' + str(i) + '.png', noncar)

    # print(len(car_features), len(notcar_features))
    # print(len(car_features), len(notcar_features))
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # print(len(X), len(y))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC(dual=False)
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

    if save_file == True:
        joblib.dump(svc, file_path)
        joblib.dump(X_scaler, scaler_file_path)
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()

    return svc, X_scaler


def load_model(file_path='./models/model.plk', scaler_file_path='./models/scaler.plk'):
    return joblib.load(file_path), joblib.load(scaler_file_path)


# TODO: Tweak these parameters and see how the results change.
color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 0  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off


svc, X_scaler = train_SVM(file_path='./models/linearModel.pkl', color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient,
                          pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

# svc, X_scaler = load_model()

# svc, X_scaler = load_model(file_path='./models/linearModel.pkl',)

cap = cv2.VideoCapture('./test_video.mp4')

last_frame = None

while(True):
    # Capture frame-by-frame
    ret, image = cap.read()

    if ret == True:

        # image = mpimg.imread('./test_images/test1.jpg')

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image.astype(np.float32)/255

        draw_image = np.copy(image)

        imshape = image.shape

        ystart = 390
        ystop = 665

        scale = 1

        hot1_windows, window1_img = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient,
                                              pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, color_space)
        # scale = 1.5
        # hot2_windows, window2_img = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient,
        #                                       pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, color_space)
        scale = 2
        hot3_windows, window3_img = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient,
                                              pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, color_space)

        hot_windows = []
        hot_windows.extend(hot1_windows)
        # hot_windows.extend(hot2_windows)
        hot_windows.extend(hot3_windows)
        window_img = draw_boxes(image, hot_windows,
                                color=(0, 0, 255), thick=6)

        cv2.imshow('Windows', window_img)

        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        # np.set_printoptions(threshold=np.nan)
        # print(box_windows)
        # Add heat to each box in box list
        heat = add_heat(heat, hot_windows)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 3)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)

        box_img = draw_labeled_bboxes(np.copy(image), labels)
        final_image = cv2.cvtColor(box_img, cv2.COLOR_RGB2BGR)

        # final_image = final_image.astype(np.float32)*255

        cv2.imshow("Final", final_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

        # fig = plt.figure()
        # plt.subplot(121)
        # plt.imshow(box_img)
        # plt.title('Car Positions')
        # plt.subplot(122)
        # plt.imshow(heatmap, cmap='hot')
        # plt.title('Heat Map')
        # fig.tight_layout()
        # plt.show()
