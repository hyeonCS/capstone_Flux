from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import threading
import socket
from copy import deepcopy

CamZone = np.array([	
	[ [2,5],[3,5],[4,5] ]	# parking camera pinumber 10's parking place x,y value
])
ParkingZone = [0, 0, 0]
prev_parkingZone = deepcopy(ParkingZone)
def genMsg(x,y,isPark,car):
	msg = str(y)+':'+str(x)+':'+str(isPark)+':'+str(car)
	return msg

def isDiffZone():
	for i in range(0, len(ParkingZone)):
		if ParkingZone[i]  !=  prev_parkingZone[i]:
			return True
	return False

def Connecting(port):
    s = socket.socket()
    host = args["host"]
    print('host, port :  ', host, port)
    s.connect((host,port))
    return s
    
def sendData(s):
    print('starting sending data')
    # y,x park, car, obstacle, led, parked
    msg=genMsg(CamZone[0, 1-1, 1], CamZone[0, 1-1, 0],ParkingZone[1-1],0)+','
    msg+=genMsg(CamZone[0, 2-1, 1], CamZone[0, 2-1, 0],ParkingZone[2-1],0)+','
    msg+=genMsg(CamZone[0, 3-1, 1], CamZone[0, 3-1, 0],ParkingZone[3-1],0)+','
    msg+=str(args["pinumber"])
    data = msg
    print('send data: %s' % data)
    byte_data = str.encode(data)
    try:
	s.sendall(byte_data)
    except:
	print('send error!!!')
    
def readData(s):
    try:
        msg = s.recv(1024)
        print('recieve data: ',msg.decode())
	data = msg.decode('utf-8')
	#print(PinZone)
        s.close()
    except socket.error as msg:
	print('read error!!!')
        sys.stderr.write('error %s' %msg[1])
        s.close()
        print('close')
        sys.exit(2)
    return msg

def commuServer():
	s = Connecting(args["port"])
        sendData(s)
        readData(s)


def start_timer(count, flag_list, index):

    count += 1
    # print(count)
    timer = threading.Timer(1, start_timer, args=[count, flag_list, index])
    timer.start()
    flag_list[index] = count

    if count == 5:
        timer.cancel()

def loc_vehicle(bbox):
	#startX = bbox[0]
	#endX = bbox[0] + bbox[2]
	(startX, startY, endX, endY) = box.astype("int")
	loc_X = int((startX + endX) / 2)
	if 0 <= loc_X <= bound_list[0]:
		loc = 1
	elif bound_list[0] <= loc_X <= bound_list[1]:
		loc = 2
	elif bound_list[1] <= loc_X:
		loc = 3

	return loc

if __name__ == '__main__':

	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--prototxt", required=True,
					help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model", required=True,
					help="path to Caffe pre-trained model")
	ap.add_argument("-c", "--confidence", type=float, default=0.4,
					help="minimum probability to filter weak detections")
	ap.add_argument("-pi", "--picamera", type=int, default=-1,
					help="whether or not the Raspberry Pi camera should be used")
	ap.add_argument("-host", "--host", required=True,
					help="server host IP address")
	ap.add_argument("-port", "--port", type=int, required=True,
					help="server host port")
	ap.add_argument("-pinumber", "--pinumber", type=int, default=10,
					help="server host port")
	args = vars(ap.parse_args())
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
			   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
			   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
			   "sofa", "train", "tvmonitor"]
	COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
	mySector = ''


	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	capture_width = 400
	bound = int(capture_width / 3)
	bound_list = (bound, bound * 2)

	park_timer_list = [[0 for i in range(0, 4)] for i in range(0, 3)]


	flag_list = [0, 0, 0, 0, 0]

	# initialize the video stream, allow the cammera sensor to warmup,
	# and initialize the FPS counter
	print("[INFO] starting video stream...")
	vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
	time.sleep(2.0)
	fps = FPS().start()

	# loop over the frames from the video stream
	while True:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		frame = vs.read()
		frame = imutils.resize(frame, width=400)

		# grab the frame dimensions and convert it to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
									 0.007843, (300, 300), 127.5)

		# pass the blob through the network and obtain the detections and
		# predictions
		net.setInput(blob)
		detections = net.forward()

		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence

			idx = int(detections[0, 0, i, 1])

			if confidence > args["confidence"] and idx == 7:
				# extract the index of the class label from the
				# `detections`, then compute the (x, y)-coordinates of
				# the bounding box for the object

				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# draw the prediction on the frame
				label = "{}: {:.2f}%".format(CLASSES[idx],
											 confidence * 100)
				cv2.rectangle(frame, (startX, startY), (endX, endY),
							  COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

				if loc_vehicle(box) == 1 and park_timer_list[0][0] == 0:
					park_timer_list[0][0] = 1
					park_timer_list[0][1] = time.time()
					start_timer(0, flag_list, 0)
				if loc_vehicle(box) == 2 and park_timer_list[1][0] == 0:
					park_timer_list[1][0] = 1
					park_timer_list[1][1] = time.time()
					start_timer(0, flag_list, 1)
				if loc_vehicle(box) == 3 and park_timer_list[2][0] == 0:
					park_timer_list[2][0] = 1
					park_timer_list[2][1] = time.time()
					start_timer(0, flag_list, 2)

				if loc_vehicle(box) == 1 and park_timer_list[0][0] == 1:
					cur_time = time.time()
					current_time = cur_time - park_timer_list[0][1]

					if abs(flag_list[0] - current_time) > 1 and not park_timer_list[0][3] == 1:
						park_timer_list[0][0] = 0
						park_timer_list[0][1] = 0

					if current_time > 5 and not park_timer_list[0][1] == 0:
						print('site 1 is obtained')
						park_timer_list[0][1] = 0
						park_timer_list[0][2] = time.time()
						park_timer_list[0][3] = 1
						ParkingZone[1-1] = 1
				if loc_vehicle(box) == 2 and park_timer_list[1][0] == 1:
					cur_time = time.time()
					current_time = cur_time - park_timer_list[1][1]

					if abs(flag_list[1] - current_time) > 1 and not park_timer_list[1][3] == 1:
						park_timer_list[1][0] = 0
						park_timer_list[1][1] = 0

					if current_time > 5 and not park_timer_list[1][1] == 0:
						print('site 2 is obtained')
						park_timer_list[1][1] = 0
						park_timer_list[1][2] = time.time()
						park_timer_list[1][3] = 1
						ParkingZone[2-1] = 1
				if loc_vehicle(box) == 3 and park_timer_list[2][0] == 1:
					cur_time = time.time()
					current_time = cur_time - park_timer_list[2][1]

					if abs(flag_list[2] - current_time) > 1 and not park_timer_list[2][3] == 1:
						park_timer_list[2][0] = 0
						park_timer_list[2][1] = 0

					if current_time > 5 and not park_timer_list[2][1] == 0:
						print('site 3 is obtained')
						park_timer_list[2][1] = 0
						park_timer_list[2][2] = time.time()
						park_timer_list[2][3] = 1
						ParkingZone[3-1] = 1

				if loc_vehicle(box) == 1 and park_timer_list[0][3] == 1:
					park_timer_list[0][2] = time.time()
				if loc_vehicle(box) == 2 and park_timer_list[1][3] == 1:
					park_timer_list[1][2] = time.time()
				if loc_vehicle(box) == 3 and park_timer_list[2][3] == 1:
					park_timer_list[2][2] = time.time()

				# if park_timer_list[0][3] == 1

			else:
				current_time = time.time()
				current_time0 = current_time - park_timer_list[0][2]
				current_time1 = current_time - park_timer_list[1][2]
				current_time2 = current_time - park_timer_list[2][2]



				if current_time0 > 5 and park_timer_list[0][3] == 1:
					print('site 1 is free')
					park_timer_list[0][0] = 0
					park_timer_list[0][1] = 0

					park_timer_list[0][3] = 0
					ParkingZone[1-1] = 0
				if current_time1 > 5 and park_timer_list[1][3] == 1:
					print('site 2 is free')
					park_timer_list[1][0] = 0
					park_timer_list[1][1] = 0

					park_timer_list[1][3] = 0
					ParkingZone[2-1] = 0
				if current_time2 > 5 and park_timer_list[2][3] == 1:
					print('site 3 is free')
					park_timer_list[2][0] = 0
					park_timer_list[2][1] = 0

					park_timer_list[2][3] = 0
					ParkingZone[3-1] = 0
		
		if isDiffZone():			
			prev_parkingZone = deepcopy(ParkingZone)
			threading._start_new_thread(commuServer,())
		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		# update the FPS counter
		fps.update()

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()

