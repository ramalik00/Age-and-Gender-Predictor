
import numpy as np
import argparse
import imutils
import time
import cv2
import os
age_list= ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)","(35-42)", "(48-53)", "(60-100)"]
gender_list=["Male","Female"]

def detect(frame,face_detect,age_detect,gender_detect):
	
	results = []
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

	face_detect.setInput(blob)
	detections=face_detect.forward()

	# iterate over the faces
	for i in range(0,detections.shape[2]):
		
		confidence=detections[0, 0, i, 2]

		if confidence > 0.5:
			
			bounding_box=detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(X_start,Y_start,X_end,Y_end) = bounding_box.astype("int")

			face = frame[Y_start:Y_end,X_start:X_end]
			face_Blob = cv2.dnn.blobFromImage(face, 1.0,(227, 227),(78.4263377603, 87.7689143744, 114.895847746),swapRB=False)

			age_detect.setInput(face_Blob)
			pred_age = age_detect.forward()
			i = pred_age[0].argmax()
			age = age_list[i]
			age_confidence = pred_age[0][i]
			
			gender_detect.setInput(face_Blob)
			pred_gender=gender_detect.forward()
			i=pred_gender[0].argmax()
			gender=gender_list[i]
			gender_confidence=pred_gender[0][i]
			r=((X_start,Y_start,X_end,Y_end),(age,age_confidence),(gender,gender_confidence))

			results.append(r)
	return results


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",type=str,default="")
ap.add_argument("-o", "--output", type=str, default="")
args = vars(ap.parse_args())

prototxtPath=os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath=os.path.sep.join(["face_detector","mobilenet_iter_73000.caffemodel"])
face_detect=cv2.dnn.readNet(prototxtPath,weightsPath)
print("FACE DETECTOR LOADED")

prototxtPath=os.path.sep.join(["age_detector", "age_deploy.prototxt"])
weightsPath=os.path.sep.join(["age_detector", "age_net.caffemodel"])
age_detect=cv2.dnn.readNet(prototxtPath,weightsPath)
print("AGE DETECTOR LOADED")

prototxtPath=os.path.sep.join(["gender_detector","deploy_gender.prototxt"])
weightsPath=os.path.sep.join(["gender_detector","gender_net.caffemodel"])
gender_detect=cv2.dnn.readNet(prototxtPath,weightsPath)
print("GENDER DETECTOR LOADED")

vs = cv2.VideoCapture(args["input"] if args["input"]!="" else 0 )
writer=None


while True:
	
	access,frame = vs.read()
	if not access:
                break
	frame = imutils.resize(frame,width=960)
	results = detect(frame,face_detect,age_detect,gender_detect)

	
	for i,(bounding_box,age_results,gender_results) in enumerate(results):
		
		(X_start,Y_start,X_end,Y_end)=bounding_box
		age,age_confidence=age_results
		gender,gender_confidence=gender_results
		
		text1 = str(age)+": {:.2f}".format((age_confidence*100))+"%"
		text2 = str(gender)+": {:.2f}".format((gender_confidence*100))+"%"

		cv2.rectangle(frame,(X_start,Y_start),(X_end,Y_end),(0, 0, 255), 2)
		cv2.putText(frame,text1,(X_start,Y_start-10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		cv2.putText(frame,text2,(X_start, Y_end+20),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if args["output"] != "" and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"],fourcc,25,
			(frame.shape[1],frame.shape[0]),True)

	if writer is not None:
		writer.write(frame)

	
	if key == ord("q"):
		break
		

cv2.destroyAllWindows()

