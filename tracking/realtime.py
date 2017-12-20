from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import sys
import socket
import threading
import RPi.GPIO as GPIO

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

Sector00 = '0:0:0:0:0:0:0'
Sector01 = '0:1:0:0:0:0:0'
Sector02 = '0:2:0:0:0:0:0'
Sector03 = '0:3:0:0:0:0:0'
Sector04 = '0:4:0:0:0:0:0'
Sector11 = '1:1:0:0:1:0:0'
Sector12 = '1:2:1:0:0:0:0'
Sector13 = '1:3:0:0:1:0:0'
Sector14 = '1:3:0:0:1:0:0'
# x: y: park: car: obstacle: led: parked

#camera's view coordinate each pi's number
CamZone = np.array([	
	[ [5,4],[4,4],[3,4],[2,4],[1,4] ],	# Num1 pi's camera
	[ [1,4],[1,3],[1,2],[1,1],[1,0] ],	# Num2
	[ [1,0],[2,0],[3,0],[4,0],[5,0] ],	# Num3
	[ [5,0],[6,0],[7,0],[8,0],[9,0] ],	# Num4
	[ [8,0],[8,1],[8,2],[8,3],[8,4] ],	# Num5 [Not Use]
	[ [8,4],[7,4],[6,4],[5,4],[4,4] ],	# Num6 [Not Use]
	[ [5,4],[5,3],[5,2],[5,1],[5,0] ],	# Num7 
])

#check recieved data whether mine or not
def isMyPosition(x, y):
    for zone in range (0, 5):
        if CamZone[PiId,zone,0] == y and CamZone[PiId,zone,1] == x:
            return True
    return False

PiId=0

def genMsg(x,y,isPark,car):
	msg = str(y)+':'+str(x)+':'+str(isPark)+':'+str(car)+','+str(args["pinumber"])
	return msg

def loc_vehicle(bbox):
        startX = bbox[0]
	endX = bbox[0] + bbox[2]
	loc_X = int((startX + endX) / 2)
	if 0 <= loc_X <= bound_list[0]:
		loc = 1
	elif bound_list[0] <= loc_X <= bound_list[1]:
		loc = 2
	elif bound_list[1] <= loc_X <= bound_list[2]:
		loc = 3
	elif bound_list[2] <= loc_X <= bound_list[3]:
		loc = 4
	elif bound_list[3] <= loc_X:
		loc = 5

	try:
		return loc
	except UnboundLocalError:
		pass


###### LED Part -----------------------------------------------
Pin1 = 11	# gpio readall / Physical number
Pin2 = 13
Pin3 = 15
Pin4 = 16
Pin5 = 18

Pin11 = 36
Pin12 = 38
Pin13 = 40

PinZone = {}

def turnOn(pin):
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.HIGH)
    
def turnOff(pin):
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)
###### LED Part END ------------------------------------------


def Connecting(port):
    s = socket.socket()
    host = args["host"]
    print('host, port :  ', host, port)
    s.connect((host,port))
    return s
    
def sendData(s,sector):
    print('starting sending data')
    # y,x park, car, obstacle, led, parked
    data = sector
    print('send data: %s' % data)
    byte_data = str.encode(data)
    s.sendall(byte_data)
    
def readData(s):
    try:
        msg = s.recv(1024)
        print('recieve data: ',msg.decode())
	data = msg.decode('utf-8')
	ledMsgs = data.split(',')
	for msg in ledMsgs:
		#print('msg:',msg)
		if len(msg)==7 and isMyPosition(int(msg[2]), int(msg[0])):
			#print('myposition',str(msg[0])+str(msg[2]))
			if int(msg[4])==1:
				print('typetype pin',PinZone[str(msg[0])+str(msg[2])] )				
				if str(msg[0])+str(msg[2])+'blue' in PinZone:
					turnOff(PinZone[str(msg[0])+str(msg[2])+'blue'] )
				turnOn(PinZone[str(msg[0])+str(msg[2])] )
			elif int(msg[4])==2:
                                print('typetype pin',PinZone[str(msg[0])+str(msg[2])+'blue'] )
				turnOff(PinZone[str(msg[0])+str(msg[2])] )
				turnOn(PinZone[str(msg[0])+str(msg[2])+'blue'] )				
			if int(msg[6]) == 1:
				turnOff(PinZone[str(msg[0])+str(msg[2])] )
				if str(msg[0])+str(msg[2])+'blue' in PinZone:
					turnOff(PinZone[str(msg[0])+str(msg[2])+'blue'] )
	#print(PinZone)
        s.close()
    except socket.error as msg:
        sys.stderr.write('error %s' %msg[1])
        s.close()
        print('close')
        sys.exit(2)
    return msg

def commuServer(sector):
	s = Connecting(args["port"])
        sendData(s,sector)
        readData(s)


def trackCar(frame, video, bbox):
	tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
	tracker_type = tracker_types[4]

	if int(minor_ver) < 3:
		tracker = cv2.Tracker_create(tracker_type)
	else:
		if tracker_type == 'BOOSTING':
			tracker = cv2.TrackerBoosting_create()
		if tracker_type == 'MIL':
			tracker = cv2.TrackerMIL_create()
		if tracker_type == 'KCF':
			tracker = cv2.TrackerKCF_create()
		if tracker_type == 'TLD':
			tracker = cv2.TrackerTLD_create()
		if tracker_type == 'MEDIANFLOW':
			tracker = cv2.TrackerMedianFlow_create()
		if tracker_type == 'GOTURN':
			tracker = cv2.TrackerGOTURN_create()

	# Initialize tracker with first frame and bounding box
	ok = tracker.init(frame, bbox)
	
	#mySector=genMsg(0,0,0,1)
	#threading._start_new_thread(commuServer,((mySector),))
	# x: y: park: car: obstacle: led: parked
	prev_loc_vec = -1
	fixCnt=0
	while True:
		frame = vs.read()
		frame = imutils.resize(frame, width=400)

		# Start timer
		timer = cv2.getTickCount()

		# Update tracker
		ok, bbox = tracker.update(frame)

		# Calculate Frames per second (FPS)
		fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
		loc_veh = None
		# Draw bounding box
		if ok:
			# Tracking success
			p1 = (int(bbox[0]), int(bbox[1]))
			p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
			cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
		else:
			# Tracking failure
			# x: y: park: car: obstacle: led: parked
			if prev_loc_vec != -1:
				print('tracking outed!')
				mySector=genMsg(CamZone[PiId, prev_loc_vec-1, 1], CamZone[PiId, prev_loc_vec-1, 0],0,0)
				threading._start_new_thread(commuServer,((mySector),))
			#cv2.destroyAllWindows()
			return

		# Display tracker type on frame
		cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50),
					2);

		# Display FPS on frame
		cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50),
					2);

		# Display result
		cv2.imshow("Frame", frame)

		loc_veh = loc_vehicle(bbox)
		#fixCnt += 1	
		#if prev_loc_vec != -1 and prev_loc_vec == loc_veh and fixCnt>250:
		#	break
		if not loc_veh == None and prev_loc_vec != loc_veh:
			yyy = CamZone[PiId, loc_veh-1, 0]
			xxx = CamZone[PiId, loc_veh-1, 1]
			mySector=genMsg(xxx,yyy,0,1)
			prev_loc_vec = loc_veh
			fixCnt=0
			print(mySector)
			threading._start_new_thread(commuServer,((mySector),))

		# Exit if ESC pressed
		k = cv2.waitKey(1) & 0xff
		if k == 27: break




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
	ap.add_argument("-pinumber", "--pinumber", type=int, default=4,
					help="server host port")
	args = vars(ap.parse_args())
	# CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	# 		   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	# 		   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	# 		   "sofa", "train", "tvmonitor"]
	CLASSES = ["background", "car"]
	COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
	mySector = ''

	PiId = args["pinumber"]-1

	PinZone[str(CamZone[PiId,0,0])+str(CamZone[PiId,0,1])] =Pin5
	PinZone[str(CamZone[PiId,1,0])+str(CamZone[PiId,1,1])]= Pin1
	PinZone[str(CamZone[PiId,2,0])+str(CamZone[PiId,2,1])]= Pin2
	PinZone[str(CamZone[PiId,3,0])+str(CamZone[PiId,3,1])]= Pin3
	PinZone[str(CamZone[PiId,4,0])+str(CamZone[PiId,4,1])]= Pin4

	PinZone[str(CamZone[PiId,1,0])+str(CamZone[PiId,1,1])+'blue']= Pin11
	PinZone[str(CamZone[PiId,2,0])+str(CamZone[PiId,2,1])+'blue']= Pin12
	PinZone[str(CamZone[PiId,3,0])+str(CamZone[PiId,3,1])+'blue']= Pin13

	print('[INFO] Pinumber: ',PiId,'  PinZone: ', PinZone)
	
	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	capture_width = 400
	bound = int(capture_width / 5)
	bound_list = (bound, bound * 2, bound * 3, bound * 4)

	# initialize the video stream, allow the cammera sensor to warmup,
	# and initialize the FPS counter
	print("[INFO] starting video stream...")
	vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
	time.sleep(2.0)
	fps = FPS().start()

	# loop over the frames from the video stream

	while True:
		car_detected = False
		frameCount = -1
		while True:
			# grab the frame from the threaded video stream and resize it
			# to have a maximum width of 400 pixels
			frame = vs.read()
			frame = imutils.resize(frame, width=400)
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break
			frameCount += 1
			if frameCount<50:
				continue


			# grab the frame dimensions and convert it to a blob
			(h, w) = frame.shape[:2]
			blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
										 0.007843, (300, 300), 127.5)
			# pass the blob through the network and obtain the detections and
			# predictions
			a = time.time()
			net.setInput(blob)
			detections = net.forward()
			# print(time.time() - a)
			for i in np.arange(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated with
				# the prediction
				confidence = detections[0, 0, i, 2]
				idx = int(detections[0, 0, i, 1])

				# filter out weak detections by ensuring the `confidence` is
				# greater than the minimum confidence
				if confidence > args["confidence"] and idx == 7:
					# extract the index of the class label from the
					# `detections`, then compute the (x, y)-coordinates of
					# the bounding box for the object

					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")

					# # draw the prediction on the frame
					# label = "{}: {:.2f}%".format(CLASSES[idx],
					# 							 confidence * 100)
					# cv2.rectangle(frame, (startX, startY), (endX, endY),
					# 			  COLORS[idx], 2)
					# y = startY - 15 if startY - 15 > 15 else startY + 15
					# cv2.putText(frame, label, (startX, y),
					# 			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
					# if CLASSES[idx] == 'car':
					# 	print(CLASSES[idx], startX, startY, endX, endY)
					# 	bbox = (startX, startY, endX - startX, endY - startY)
					# 	car_detected = True
					print('car', startX, startY, endX, endY)
					bbox = (startX, startY, endX - startX, endY - startY)
					car_detected = True
				else:
					break
			# show the output frame
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				GPIO.cleanup()
				exit(0)

			# update the FPS counter
			fps.update()
			if car_detected == True:
				break
			else:
				yyy = CamZone[PiId, 2, 0]
				xxx = CamZone[PiId, 2, 1]
				mySector=genMsg(0,0,0,0)
				#int(mySector)
				threading._start_new_thread(commuServer,((mySector),))
		
		# # stop the timer and display FPS information
		fps.stop()
		print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
		print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
		trackCar(frame, vs, bbox)
		
		# # do a bit of cleanup
		# cv2.destroyAllWindows()
		# vs.stop()

