# USAGE
"""
python recognize_video.py --detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7 \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle
"""

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
from scipy.ndimage.filters import gaussian_filter
from PIL import Image, ImageChops 

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()
face_colors_init = (0,0,0)
bg_colors_init = (0,0,0)
frame_init = 0
frames = []


# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()

	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

			# draw the bounding box of the face along with the
			# associated probability
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

			bg = frame.copy()
			cv2.rectangle(bg, (startX, startY), (endX, endY), (0,0,0), -1)

			if np.all(face_colors_init) == 0:
				face_colors_init = np.mean(face, axis=(0, 1))
				bg_colors_init = np.sum(bg, axis=(0, 1))/(bg.shape[0]*bg.shape[1] - face.shape[0]*face.shape[1])
				frame_init = frame.copy()
			
			frames.append(frame)
			if len(frames) > 40:
				# print(np.array(frames).shape)
				del(frames[0])
				# frame = np.mean(frames, axis=(0))
				# print(np.array(frame).shape)

			face_colors = np.mean(face, axis=(0, 1))
			bg_colors = np.sum(bg, axis=(0, 1))/(bg.shape[0]*bg.shape[1] - face.shape[0]*face.shape[1])

			# print
			print(face.shape, type(face), np.subtract(face_colors, face_colors_init), np.subtract(bg_colors, bg_colors_init))

	# update the FPS counter
	fps.update()

	# show the output frame
	#comment out next 2 lines for normal stream
	# if len(frames) > 5:
	# 	frame = np.average(frames[0:5], axis=(0), weights=np.array([1,2,3,4,5], dtype=np.uint8))
	# frame[frame < 0] *= -1
	# print(frame.shape)
	
	# frame[frame < 50] = 0
	# frame[frame > 200] = 0
	# kernel = np.ones((5, 5), np.uint8)
	# cv2.dilate(frame, kernel, iterations = 1)
	# frame = np.absolute(frame)
	# blurred = gaussian_filter(frame, sigma=1)
	# frame = cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,21)
	# np.absolute(frame)

	# frame = frames[0] - frames[-1]
	frame = ImageChops.subtract(Image.fromarray(frames[-1]), Image.fromarray(frame_init), scale=0.25)
	# print(np.array(frames)[0].shape, '$')
	cv2.imshow("Frame", np.array(frame))
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()