# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import time
import cv2
import os

def detect_and_predict_mask(image, faceNet, maskNet, args):
    # grab the dimensions of the frame and then construct a blob
    # from it
    if img is None:
        break
    orig = image.copy()
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args.confidence:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face
            # has a mask or not
            (withMask, withMaskIncorrect, withoutMask) = maskNet.predict(face)[0]

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if withMask > withoutMask and withMask > withMaskIncorrect else "Incorrect Mask" if withMaskIncorrect > withMask and withMaskIncorrect > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 255, 255) if label == "Incorrect Mask" else (0,0,255) 

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(withMask, withoutMask, withMaskIncorrect) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(image, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    cv2.namedWindow("output", cv2.WINDOW_NORMAL)  
    cv2.imwrite("Output/output.jpg", image)
    cv2.imshow("output",image)
    cv2.waitKey(0)

def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, 
    help="path to input image")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
    args = ap.parse_args()

    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.join(os.getcwd(), "deploy.prototxt") 
    weightsPath = os.path.join(os.getcwd(),"res10_300x300_ssd_iter_140000.caffemodel")
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    maskModelPath = os.path.join(os.getcwd(), "mask-detector-model.model")
    model = load_model(maskModelPath)

    # load the input image from disk, clone it, and grab the image spatial
    # dimensions
    image = cv2.imread(args.image)

    detect_and_predict_mask(image, net, model, args)

if __name__ == '__main__':
    main()