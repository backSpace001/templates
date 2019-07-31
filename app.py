from flask import Flask
from flask import Flask, render_template,Response,request ,make_response, session   
import pandas as pd
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from darkflow.net.build import TFNet
import numpy as np
import label_image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2
import pytesseract
from PIL import Image
import random
import pickle
import os
from os.path import isfile, join
app = Flask(__name__,static_folder = "templates")

#**************** Modify This Functions ****************#

def processFaceAndEyeImage(imagePath): #You can see the function name here. It is for Face & Eyye image.
	#read images from directories
	img = cv2.imread(imagePath,0)

	# Trained XML classifiers describes some features of some 
	# object we want to detect a cascade function is trained 
	# from a lot of positive(faces) and negative(non-faces) 
	# images. 
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
	def detect_face(img):
		face_img = img.copy()
		face_rects = face_cascade.detectMultiScale(face_img)
		for(x,y,w,h) in face_rects:
			cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),5)
			return face_img

	result1 = detect_face(img)


	eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
	def detect_eyes(img):
		face_img = img.copy()
		eyes_rects = eye_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5)
		for(x,y,w,h) in eyes_rects:
			cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),2)
			return face_img

	result2 = detect_eyes(img)
	img1 = "result1face.jpg"
	img2 = "result2eye.jpg"
	cv2.imwrite('templates/'+img1,result1)
	cv2.imwrite('templates/'+img2,result2)
	return "<img src='templates/"+img1+"?="+str(random.randint(0,100000000000000000000))+"'></img><br><img src='templates/"+img2+"'></img><br><br><a href='http://127.0.0.1:5000/'>Home</a>"
	#The function will return above line and show it to user.
	

def processFaceAndEyeVideo(videoPath):

	# Trained XML classifiers describes some features of some 
	# object we want to detect a cascade function is trained 
	# from a lot of positive(faces) and negative(non-faces) 
	# images. 
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
	def detect_face(img):
		face_img = img.copy()
		face_rects = face_cascade.detectMultiScale(face_img)
		for(x,y,w,h) in face_rects:
			cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
			return face_img

	eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
	def detect_eyes(img):
		face_img = img.copy()
		eyes_rects = eye_cascade.detectMultiScale(face_img)
		for(x,y,w,h) in eyes_rects:
			cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
			return face_img


	cap = cv2.VideoCapture(videoPath) 
	
	codecformat = cv2.VideoWriter_fourcc(*'XVID')
	size = (
		int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
		int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	)
	out = cv2.VideoWriter('templates/faceandeyeVideo.avi',codecformat, 20.0, size)
	# loop runs if capturing has been initialized. 
	while 1:  
	  
		# reads frames from a camera 
		ret, img = cap.read()  
		if ret == False:
			break
		# convert to gray scale of each frames 
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
	  
		# Detects faces of different sizes in the input image 
		faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
	  
		for (x,y,w,h) in faces: 
			# To draw a rectangle in a face  
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  
			roi_gray = gray[y:y+h, x:x+w] 
			roi_color = img[y:y+h, x:x+w] 
	  
			# Detects eyes of different sizes in the input image 
			eyes = eye_cascade.detectMultiScale(roi_gray)  
	  
			#To draw a rectangle in eyes 
			for (ex,ey,ew,eh) in eyes: 
				cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2) 

		out.write(img)
		
	#cv2.imshow('frame',frame)
	#convert_frames_to_video('templates/faceandeyeVideo/', 'templates/faceandeyeVideo.avi', 24)
	return "<a style='font-size:35px;font-weight:900;text-align:center;' href='templates/faceandeyeVideo.avi' download>Download Video</a><br><br><a href='http://127.0.0.1:5000/'>Home</a>"
	#This is the code for face and image video. You need to change this...just like this...for all functions



def processCelebrityImage(imagePath,gender):

	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

	recognizer = cv2.face.LBPHFaceRecognizer_create()
	if gender == "male":
		recognizer.read("male.yml")
		print("Male")
	elif gender == "female":
		recognizer.read("female.yml")
		print("Female")
	else:
		return "Something went wrong"

	labels = {"person_name": 1}
	with open("labels.pickle", 'rb') as f:
		og_labels = pickle.load(f)
		labels = {v:k for k,v in og_labels.items()}


	frame = cv2.imread(imagePath)

	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	for (x,y,w,h) in faces:
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = frame[y:y+h, x:x+w]
			id_, conf = recognizer.predict(roi_gray)
			if conf>=45 and conf <=85:
				print(id_)
				print(labels[id_])
				font = cv2.FONT_HERSHEY_SIMPLEX
				name = labels[id_]
				color = (255, 255, 255)
				cv2.putText(frame, name, (x,y), font, 1, color, 2, cv2.LINE_AA)
			img_item = "my-image.png"
			cv2.imwrite(img_item, roi_color)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),5)
			print(id_)
			print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255, 255, 255)
			cv2.putText(frame, name, (x,y), font, 1, color, 2, cv2.LINE_AA)
			cv2.imwrite('templates/'+img_item,frame)
	if len(faces) == 0:
		return "No Face Found"
	return "<img src='templates/"+img_item+"?="+str(random.randint(0,100000000000000000000))+"'></img><br><br><a href='http://127.0.0.1:5000/'>Home</a>"



def processCelebrityVideo(videoPath, gender):

	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	if gender == "male":
		recognizer.read("male.yml")
	elif gender == "female":
		recognizer.read("female.yml")
	else:
		return "Something went wrong"
	labels = {"person_name": 1}

	with open("labels.pickle", 'rb') as f:
		og_labels = pickle.load(f)
		labels = {v:k for k,v in og_labels.items()}


	cap = cv2.VideoCapture(videoPath)

	codecformat = cv2.VideoWriter_fourcc(*'XVID')
	size = (
		int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
		int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	)
	out = cv2.VideoWriter('templates/celebrityVideo.avi',codecformat, 20.0, size)
	while (True):
		
		
		ret,frame = cap.read()
		if ret==False:
			break
	#		frame = cv2.flip(frame,0)
	#	else:
	#		break
		print(ret)
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
		print(faces)
		for (x,y,w,h) in faces:
			#print(x,y,w,h)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = frame[y:y+h, x:x+w]
			id_, conf = recognizer.predict(roi_gray)
			if conf>=45 and conf <=85:
				print(id_)
				print(labels[id_])
				font = cv2.FONT_HERSHEY_SIMPLEX
				name = labels[id_]
				color = (255, 255, 255)
				stroke = 2
				cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
				
			

			
			color = (255, 0, 0)
			stroke = 2
			end_cord_x = x + w
			end_cord_y = y + h
			cv2.rectangle(frame,(x, y), (end_cord_x, end_cord_y), color, stroke)
			out.write(frame)
			
		#cv2.imshow('frame',frame)
	#convert_frames_to_video('templates/celebrityVideo/', 'templates/celebrityVideo.avi', 24)
	return "<a style='font-size:35px;font-weight:900;text-align:center;' href='templates/celebrityVideo.avi' download>Download Video</a> <br><br><a href='http://127.0.0.1:5000/'>Home</a></button>"



def processObjectImage(imagePath):
	#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
	options = {
		'model': 'cfg/yolo.cfg',
		'load': 'bin/yolov2.weights',
		'threshold': 0.3,
		'gpu': 1.0
	}
	tfnet = TFNet(options)
	img = cv2.imread(imagePath, cv2.IMREAD_COLOR)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# use YOLO to predict the image
	result = tfnet.return_predict(img)

	tl = (result[0]['topleft']['x'], result[0]['topleft']['y'])
	br = (result[0]['bottomright']['x'], result[0]['bottomright']['y'])
	label = result[0]['label']


	# add the box and label and display it
	img = cv2.rectangle(img, tl, br, (0, 255, 0), 7)
	img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
	
	cv2.imwrite('templates/object_detection.jpg',img)
	return "<img src='templates/object_detection.jpg?="+str(random.randint(0,100000000000000000000))+"'></img><br><br><button style='font-size:20px;font-weight:900;color:black;background-color:lightblue;border:0;padding:20px 10px;'><a href='http://127.0.0.1:5000/'>Home</a></button>"
			

def processObjectVideo(videoPath):
	option = {
		'model': 'cfg/yolo.cfg',
		'load': 'bin/yolov2.weights',
		'threshold': 0.15,
		'gpu': 1.0
	}

	tfnet = TFNet(option)

	capture = cv2.VideoCapture(videoPath)
	colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
	size = (
		int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
		int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	)
	codecformat = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('templates/objectoutput.avi',codecformat, 20.0, size)

	while (capture.isOpened()):
		ret, frame = capture.read()
		if ret:
			results = tfnet.return_predict(frame)
			for color, result in zip(colors, results):
				tl = (result['topleft']['x'], result['topleft']['y'])
				br = (result['bottomright']['x'], result['bottomright']['y'])
				label = result['label']
				frame = cv2.rectangle(frame, tl, br, color, 7)
				frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
				
			out.write(frame)
		else:
			break
	return "<a style='font-size:35px;font-weight:900;text-align:center;' href='templates/objectoutput.avi' download>Download Video</a><br><br><button style='font-size:20px;font-weight:900;color:black;background-color:lightblue;border:0;padding:20px 10px;'><a href='http://127.0.0.1:5000/'>Home</a></button> "


def processReadTextImage(imagePath):
	img = Image.open(imagePath)
	pytesseract.pytesseract.tesseract_cmd = 'tesseract'
	result = pytesseract.image_to_string(img)
	return result + "<br><br><button style='font-size:20px;font-weight:900;color:black;background-color:lightblue;border:0;padding:20px 10px;'><a href='http://127.0.0.1:5000/'>Home</a></button>"

def processFacialImage(imagePath):
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

	recognizer = cv2.face.LBPHFaceRecognizer_create()
	recognizer.read("face-trainner2.yml")

	labels = {"person_name": 1}
	with open("labels.pickle", 'rb') as f:
		og_labels = pickle.load(f)
		labels = {v:k for k,v in og_labels.items()}


	frame = cv2.imread(imagePath)

	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	for (x,y,w,h) in faces:
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = frame[y:y+h, x:x+w]
			id_, conf = recognizer.predict(roi_gray)
			if conf>=45 and conf <=85:
				print(id_)
				print(labels[id_])
				font = cv2.FONT_HERSHEY_SIMPLEX
				name = labels[id_]
				color = (255, 255, 255)
				cv2.putText(frame, name, (x,y), font, 1, color, 2, cv2.LINE_AA)
			img_item = "my-image.png"
			cv2.imwrite(img_item, roi_color)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),5)
			print(id_)
			print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255, 255, 255)
			cv2.putText(frame, name, (x,y), font, 1, color, 2, cv2.LINE_AA)
	
	cv2.imwrite('templates/'+img_item,frame)
	if len(faces) == 0:
		return "No Face Found"
	return "<img src='templates/"+img_item+"?="+str(random.randint(0,100000000000000000000))+"'></img><br><br><button style='font-size:20px;font-weight:900;color:black;background-color:lightblue;border:0;padding:20px 10px;'><a href='http://127.0.0.1:5000/'>Home</a></button>"
			
			
def processFacialVideo(videoPath):
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	recognizer.read("face-trainner.yml")
	labels = {"person_name": 1}

	with open("labels.pickle", 'rb') as f:
		og_labels = pickle.load(f)
		labels = {v:k for k,v in og_labels.items()}

	cap = cv2.VideoCapture(videoPath)
	codecformat = cv2.VideoWriter_fourcc(*'XVID')
	size = (
		int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
		int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	)
	out = cv2.VideoWriter('templates/facialVideo.avi',codecformat, 20.0, size)
	while (True):
		
		ret,frame = cap.read()
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
		for (x,y,w,h) in faces:
			#print(x,y,w,h)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = frame[y:y+h, x:x+w]
			id_, conf = recognizer.predict(roi_gray)
			if conf>=45 and conf <=85:
				print(id_)
				print(labels[id_])
				font = cv2.FONT_HERSHEY_SIMPLEX
				name = labels[id_]
				color = (255, 255, 255)
				stroke = 2
				cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
				
			
			#img_item = "my-image.png"
			#cv2.imwrite(img_item, roi_gray)
			
			color = (255, 0, 0)
			stroke = 2
			end_cord_x = x + w
			end_cord_y = y + h
			cv2.rectangle(frame,(x, y), (end_cord_x, end_cord_y), color, stroke)
			out.write(frame)
	
	#convert_frames_to_video('templates/facialVideo/', 'templates/facialVideo.avi', 24)
	return "<a style='font-size:35px;font-weight:900;text-align:center;' href='templates/facialVideo.avi' download>Download Video</a><br><br><button style='font-size:20px;font-weight:900;color:black;background-color:lightblue;border:0;padding:20px 10px;'><a href='http://127.0.0.1:5000/'>Home</a></button> "

	
		
		
def processFacialErImage(imagePath):
	size = 4

	# We load the xml file
	classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

	im = cv2.imread(imagePath, 0 )
	#im=cv2.flip(im,1,0) #Flip to act as a mirror

	# Resize the image to speed up detection
	mini = cv2.resize(im, (int(im.shape[1]/size), int(im.shape[0]/size)))

	# detect MultiScale / faces 
	faces = classifier.detectMultiScale(mini)

	# Draw rectangles around each face
	for f in faces:
		(x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
		cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 4)
			
		#Save just the rectangle faces in SubRecFaces
		sub_face = im[y:y+h, x:x+w]

		FaceFileName = "test.jpg" #Saving the current image for testing.
		#cv2.imwrite(FaceFileName, sub_face)
			
		text = label_image.main(FaceFileName)# Getting the Result from the label_image file, i.e., Classification Result.
		text = text.title()# Title Case looks Stunning.
		font = cv2.FONT_HERSHEY_TRIPLEX
		cv2.putText(im, text,(x,y), font, 1, (255,0,0), 2)
	if len(faces) == 0:
		return "No Face Found"
	cv2.imwrite('templates/'+FaceFileName,im)
	return "<img src='templates/"+FaceFileName+"?="+str(random.randint(0,100000000000000000000))+"'></img><br><br><button style='font-size:20px;font-weight:900;color:black;background-color:lightblue;border:0;padding:20px 10px;'><a href='http://127.0.0.1:5000/'>Home</a></button>"

def processActivityImage(videoPath):
	#imagePath Contains the path of the image file, you can read it
	#Process it 
	#And return the output in text (If the output is something other than text, let me know
	return "Activity Image Output" #Change this to your output

def processActivityVideo(videoPath):
	#videoPath Contains the path of the video file, you can read it
	#Process it 
	#And return the output in text (If the output is something other than text, let me know
	return "Activity Video Output" #Change this to your output



#******************Flask App Code Starts ****************
@app.route('/')
def index():
		return render_template('/index.html')

@app.route('/imageorvideo.html')
def imageorvideo():
		dowhat = request.args.get('dowhat')


		if dowhat == "celebrity":
			maleorfemale = "yes"
			return render_template('/imageorvideo.html',dowhat=dowhat,maleorfemale=maleorfemale)
		else:
			return render_template('/imageorvideo.html',dowhat=dowhat,maleorfemale="") #Here we are passing a message variable. You can customize it like this

@app.route('/faceandeye.html', methods = ['POST', 'GET'])
def faceandeye():
	if request.method == 'POST':
		f = request.files['fileToUpload']
		filePath = f.filename
		f.save(secure_filename(filePath))
		extension = filePath.split(".")
		extension = extension[len(extension)-1]
		if "jpeg" in extension or "jpg" in extension or "png" in extension:
			output = processFaceAndEyeImage(filePath)
			return output #render_template('/faceandeye.html',output=output)
		elif "mp4" in extension or "wmv" in extension or "mkv" in extension or "webm" in extension or "avi" in extension:
			output = processFaceAndEyeVideo(filePath)
			return output #render_template('/faceandeye.html',output=output)
		else:
			return "Invalid File uploaded"	
	else:
		return render_template('/index.html')


@app.route('/celebrity.html', methods = ['POST', 'GET'])
def celebrity():
	if request.method == 'POST':
		f = request.files['fileToUpload']
		filePath = f.filename
		f.save(secure_filename(filePath))
		extension = filePath.split(".")
		extension = extension[len(extension)-1]
		
		
		if "jpeg" in extension or "jpg" in extension or "png" in extension:
			output = processCelebrityImage(filePath,request.form['gender'])
			return output#render_template('/celebrity.html',output=output)
		elif "mp4" in extension or "wmv" in extension or "mkv" in extension or "webm" in extension or "avi" in extension:
			output = processCelebrityVideo(filePath,request.form['gender'])
			return output #render_template('/celebrityVideo.html')
			#return redirect('/templates/celebrityVideo.mp4')
		else:
			return "Invalid File uploaded"	
	else:
		return render_template('/index.html')

@app.route('/object.html', methods = ['POST', 'GET'])
def object():
	if request.method == 'POST':
		f = request.files['fileToUpload']
		filePath = f.filename
		f.save(secure_filename(filePath))
		extension = filePath.split(".")
		extension = extension[len(extension)-1]
		if "jpeg" in extension or "jpg" in extension or "png" in extension:
			output = processObjectImage(filePath)
			return output#render_template('/object.html',output=output)
		elif "mp4" in extension or "wmv" in extension or "mkv" in extension or "webm" in extension or "avi" in extension:
			output = processObjectVideo(filePath)
			return output#render_template('/object.html',output=output)
		else:
			return "Invalid File uploaded"	
	else:
		return render_template('/index.html')

@app.route('/readtext.html', methods = ['POST', 'GET'])
def readtext():
	if request.method == 'POST':
		f = request.files['fileToUpload']
		filePath = f.filename
		f.save(secure_filename(filePath))
		extension = filePath.split(".")
		extension = extension[len(extension)-1]
		if "jpeg" in extension or "jpg" in extension or "png" in extension:
			output = processReadTextImage(filePath)
			return output #render_template('/readtext.html',output=output)
		else:
			return "Invalid File uploaded"	
	else:
		return render_template('/index.html')
@app.route('/facial.html', methods = ['POST', 'GET'])
def facial():
	if request.method == 'POST':
		f = request.files['fileToUpload']
		filePath = f.filename
		f.save(secure_filename(filePath))
		extension = filePath.split(".")
		extension = extension[len(extension)-1]
		if "jpeg" in extension or "jpg" in extension or "png" in extension:
			output = processFacialImage(filePath)
			return output #render_template('/facial.html',output=output)
		elif "mp4" in extension or "wmv" in extension or "mkv" in extension or "webm" in extension or "avi" in extension:
			output = processFacialVideo(filePath)
			return output #render_template('/facial.html',output=output)
		else:
			return "Invalid File uploaded"	
	else:
		return render_template('/index.html')
@app.route('/facialer.html', methods = ['POST', 'GET'])
def facialer():
	if request.method == 'POST':
		f = request.files['fileToUpload']
		filePath = f.filename
		f.save(secure_filename(filePath))
		extension = filePath.split(".")
		extension = extension[len(extension)-1]
		if "jpeg" in extension or "jpg" in extension or "png" in extension:
			output = processFacialErImage(filePath)
			return output #render_template('/facial.html',output=output)
		else:
			return "Invalid File uploaded"	
	else:
		return render_template('/index.html')
@app.route('/activity.html', methods = ['POST', 'GET'])
def activity():
	if request.method == 'POST':
		f = request.files['fileToUpload']
		filePath = f.filename
		f.save(secure_filename(filePath))
		extension = filePath.split(".")
		extension = extension[len(extension)-1]
		if "jpeg" in extension or "jpg" in extension or "png" in extension:
			output = processActivityImage(filePath)
			return render_template('/activity.html',output=output)
		elif "mp4" in extension or "wmv" in extension or "mkv" in extension or "webm" in extension or "avi" in extension:
			output = processActivityVideo(filePath)
			return render_template('/activity.html',output=output)
		else:
			return "Invalid File uploaded"	
	else:
		return render_template('/index.html')

if __name__ == "__main__":
    app.run(debug=True)
    #port = int(os.environ.get("PORT", 5000))
    #app.run(host='0.0.0.0', port=port) 