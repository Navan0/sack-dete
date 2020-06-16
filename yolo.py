import cv2
import numpy as np
import argparse
import time

count = 0
def load_yolo():
	net = cv2.dnn.readNet("yolo-tiny_10000.weights", "yolo-tiny.cfg")
	classes = []
	with open("obj.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]

	layers_names = net.getLayerNames()
	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers


def display_blob(blob):
	'''
		Three images each for RED, GREEN, BLUE channel
	'''
	for b in blob:
		for n, imgb in enumerate(b):
			cv2.imshow(str(n), imgb)

def detect_objects(img, net, outputLayers):
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids

def draw_labels(boxes, confs, colors, class_ids, classes, img):
	print("d")

	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	height,width = img.shape[:2]
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			color = colors[0]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
			ymin, xmin, ymax, xmax = boxes[i]
			coord = (ymin, xmin, ymax, xmax)
			centerCoord = (coord[0]+(coord[2]/2), coord[1]+(coord[3]/2))
			# print(len(boxes))
			# count = count+1

			print(sum(centerCoord), height-400 // 1)
			cv2.circle(img, (int(centerCoord[0]),int(centerCoord[1])), radius=100, color=(0, 255, 0), thickness=-1)
			if  centerCoord[1] == height-400 // 1:
				print("d")

	cv2.line(img, (0, height-400 // 1), (width, height-400 // 1), (255, 255, 0), 7)
	cv2.imshow("Image", cv2.resize(img, (int(720),int(640))))

def webcam_detect():
	model, classes, colors, output_layers = load_yolo()
	cap = start_webcam()
	while True:
		_, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)

		# cv2.circle(image_np, (centerCoord), radius=100, color=(0, 255, 0), thickness=-1)
		# print(centerCoord)

		draw_labels(boxes, confs, colors, class_ids, classes, frame)
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()



def start_video(video_path):
	model, classes, colors, output_layers = load_yolo()
	cap = cv2.VideoCapture(video_path)
	re = 1
	cu = [0]
	while True:

		_, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
		count = draw_labels(boxes, confs, colors, class_ids, classes, frame)
		cv2.putText(frame, "COUNT =" + str(count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,2, (0, 0, 255), 3, cv2.LINE_AA)
		# if boxes:
		# 	re = re+1
		# 	cu = cu.append(boxes[0])
		#
		# if len(boxes) == 0:
		# 	print(re/len(cu))
		#
		#
		# print(re,cu)
		# cv2.line(frame, (0, height-100-1 // 1), (width, height-100-1 // 1), (255, 255, 0), 7)
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()



if __name__ == '__main__':
	video_path = "/home/navan/mine/sack_counter/test.mp4"
	start_video(video_path)

	cv2.destroyAllWindows()
