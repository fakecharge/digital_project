import os
import keras
import numpy as np
import cv2
import pandas as pd
from keras import backend as K
from keras.models import model_from_json
import json
import matplotlib.pyplot as plt

K.set_image_dim_ordering('tf')


def decode_nn_res(res_vec, num_digits, num_classes, dummy_class):
    digits = np.array_split(res_vec, num_digits)
    actual_digits = np.argmax(digits, 1) + 1
    res = actual_digits[actual_digits != dummy_class]
    return actual_digits, ''.join(map(str, res))


def process_labels(labels, max_digits):
    tmp = []
    for label in labels:
        vec = [int(float(x)) for x in label.split('_')]
        if len(vec) < max_digits:
            vec = vec + [11] * (max_digits - len(vec))
        tmp.append(vec)
    labels = np.array(tmp)
    tmp = []
    num_classes = 11
    for target in labels[:, ...]:
        y = np.zeros((len(target), num_classes))
        y[np.arange(target.shape[0]), target - 1] = 1
        tmp.append(y)
    labels = np.array(tmp)
    return labels





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~load models~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('loading cnn models')

# load digit detection model
with open(os.path.join('model', 'digit_detection_cnn_layers.json'), 'r') as json_data:
    model_dict = json.load(json_data)

detect_model = model_from_json(json.dumps(model_dict))
detect_model.load_weights(os.path.join('model', 'digit_detection_cnn_weights.h5'))

# load digit classification model
with open(os.path.join('model', 'digit_classification_cnn_layers.json'), 'r') as json_data:
    model_dict = json.load(json_data)

classification_model = keras.models.model_from_json(json.dumps(model_dict))
classification_model.load_weights(os.path.join('model', 'digit_classification_cnn_weights.h5'))

print('single image prediction results')


def find_box_and_predict_digit(input_img):
    num_digits = 4
    input_img_shape = input_img.shape
    train_img_size = (64, 64)
    proc_input_img = np.array(
        cv2.normalize(cv2.cvtColor(cv2.resize(input_img, train_img_size), cv2.COLOR_BGR2GRAY).astype(np.float64), 0, 1,
                      cv2.NORM_MINMAX)[..., np.newaxis])[np.newaxis, ...]
    box_preds = detect_model.predict(proc_input_img)
    scaled_box = box_preds[0].copy()
    scaled_box[0] = scaled_box[0] / float(train_img_size[0] / input_img_shape[0])
    scaled_box[1] = scaled_box[1] / float(train_img_size[1] / input_img_shape[1])
    scaled_box[2] = scaled_box[2] / float(train_img_size[1] / input_img_shape[1])
    scaled_box[3] = scaled_box[3] / float(train_img_size[0] / input_img_shape[0])
    start_row = np.clip(int(scaled_box[0]), 1, input_img_shape[0])
    end_row = np.clip(int(scaled_box[0] + scaled_box[3]), 1, input_img_shape[0])
    start_col = np.clip(int(scaled_box[1]), 1, input_img_shape[1])
    end_col = np.clip(int(scaled_box[1] + scaled_box[2]), 1, input_img_shape[1])
    # need better logic to handle cases where the box is too thin
    if start_col - end_col == 0:
        start_col -= 1
    if start_row - end_row == 0:
        start_row -= 1
    # store only the cutouts
    digits_only = input_img[start_row:end_row, start_col:end_col, ...]
    digits_only_resized = cv2.resize(digits_only, train_img_size)
    orig_img_box = input_img.copy()
    cv2.rectangle(orig_img_box, (start_col, start_row), (end_col, end_row), (0, 255, 0), 1)
    plt.imshow(orig_img_box)
    plt.show()
    digit_pred = classification_model.predict(np.array(digits_only_resized)[np.newaxis, ...])
    score = np.concatenate(digit_pred, axis=1)
    pred_labels_encoded = np.zeros(score.shape, dtype="int32")
    pred_labels_encoded[score > 0.5] = 1
    pred_labels_decoded = np.array([decode_nn_res(x, num_digits, 11, 11) for x in pred_labels_encoded])
    pred_labels_decoded_digits = np.array(pred_labels_decoded[:, 1])
    # pred_labels_decoded_OHE_digits = np.array(pred_labels_decoded[:,0])
    final_digit = pred_labels_decoded_digits[0]
    print('Predicted digit:', final_digit)
    return final_digit

#Labels of network.
classNames = {0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

img = cv2.imread("test_img/test8.jpg")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
frame_resized = cv2.resize(img, (800, 600))  # resize frame for prediction
blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
net.setInput(blob)
detections = net.forward()

cols = frame_resized.shape[1]
rows = frame_resized.shape[0]

for i in range(detections.shape[2]):
    class_id = int(detections[0, 0, i, 1])
    if not class_id == 6:
        continue
    confidence = detections[0, 0, i, 2]
    if confidence > 0.3:
        class_id = int(detections[0, 0, i, 1])
        xLeftBottom = int(detections[0, 0, i, 3] * cols)
        yLeftBottom = int(detections[0, 0, i, 4] * rows)
        xRightTop = int(detections[0, 0, i, 5] * cols)
        yRightTop = int(detections[0, 0, i, 6] * rows)

        heightFactor = img.shape[0] / 600.0
        widthFactor = img.shape[1] / 800.0

        xLeftBottom = int(widthFactor * xLeftBottom)
        yLeftBottom = int(heightFactor * yLeftBottom)
        xRightTop = int(widthFactor * xRightTop)
        yRightTop = int(heightFactor * yRightTop)

        cv2.rectangle(img, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                      (0, 255, 0))
        roi = img[yLeftBottom:yRightTop, xLeftBottom:xRightTop]
        cv2.imshow("roi", roi)
        new_roi = roi.copy()
        new_roi = cv2.cvtColor(new_roi, cv2.COLOR_BGR2GRAY)
        th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        #find_box_and_predict_digit(roi)


        if class_id in classNames:
            label = classNames[class_id] + ": " + str(confidence)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            yLeftBottom = max(yLeftBottom, labelSize[1])
            cv2.rectangle(img, (xLeftBottom, yLeftBottom - labelSize[1]),
                          (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(img, label, (xLeftBottom, yLeftBottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            print(label)

