
import numpy as np
import argparse
import cv2
import os


ap=argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True)

args=vars(ap.parse_args())


age_list=["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)","(38-43)", "(48-53)", "(60-100)"]
gender_list=['Male','Female']

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


image=cv2.imread(args["input"])
(h,w)=image.shape[:2]
blob=cv2.dnn.blobFromImage(image,1.0,(500,500),(104.0,177.0,123.0))

face_detect.setInput(blob)
detections=face_detect.forward()

# iterate over the faces
for i in range(0,detections.shape[2]):
	
	confidence = detections[0, 0, i, 2]

	if confidence > 0.5:
		
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(X_start,Y_start,X_end,Y_end)=box.astype("int")

		face = image[Y_start:Y_end, X_start:X_end]
		face_Blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),(78.4263377603, 87.7689143744, 114.895847746),swapRB=False)

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
		
		text1 = str(age)+": {:.2f}".format((age_confidence*100))+"%"
		text2 = str(gender)+": {:.2f}".format((gender_confidence*100))+"%"

		cv2.rectangle(image,(X_start,Y_start),(X_end,Y_end),(0, 0, 255), 2)
		cv2.putText(image,text1,(X_start,Y_start-10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		cv2.putText(image, text2,(X_start, Y_end+20),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


cv2.imshow("Frame", image)
cv2.waitKey(0)
