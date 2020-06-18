"""
 * Copyright (C) 2020 Navaneeth KT nktclt@gmail.com>
 * This file is part of NOUS_BLOC#21.
 * It can not be copied and/or distributed without the express
 * permission of Navaneeth KT
"""




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



def start_video(video_path):
	model, classes, colors, output_layers = load_yolo()
	cap = cv2.VideoCapture(video_path)
	re = True
	count = 0
	tot = []
	centerCoord = (0,0)
	a = {}
	key =0
	while True:

		_, frame = cap.read()
		height, width, channels = frame.shape
		blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
		model.setInput(blob)
		outputs = model.forward(output_layers)
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
		indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
		font = cv2.FONT_HERSHEY_PLAIN
		height,width = frame.shape[:2]
		if len(boxes) == 0:
			re = True
		for i in range(len(boxes)):

			# print(len(boxes))
			if i in indexes:
				# print(indexes,len(boxes))
				x, y, w, h = boxes[i]
				label = str(classes[class_ids[i]])
				color = colors[0]
				cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
				cv2.putText(frame, label, (x, y - 5), font, 10, color, 1)
				ymin, xmin, ymax, xmax = boxes[i]
				coord = (ymin, xmin, ymax, xmax)
				centerCoord = (coord[0]+(coord[2]/2), coord[1]+(coord[3]/2))
				a[i] = centerCoord
				# tot.append(centerCoord)
				# import pdb; pdb.set_trace()
				cv2.circle(frame, (int(centerCoord[0]),int(centerCoord[1])), 7, (255, 255, 255), -1)
				if a[x][1] > height-400+100 // 1 :
					if a[x][1] > height-390+100 // 1 and re == True:
						# print("sin",len(tot))
						count = count+1
						re = False


		cv2.line(frame, (0, height-400+100 // 1), (width, height-400+100 // 1), (255, 0, 0), 7)
		cv2.line(frame, (0, height-390+100 // 1), (width, height-390+100 // 1), (255, 255, 0), 7)
		cv2.putText(frame, "COUNT =" + str(count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,2, (0, 0, 255), 3, cv2.LINE_AA)
		cv2.imshow("Image", cv2.resize(frame, (int(720),int(640))))
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()



if __name__ == '__main__':
	video_path = "/home/navan/mine/sack_counter/test.mp4"
	start_video(video_path)

	cv2.destroyAllWindows()
