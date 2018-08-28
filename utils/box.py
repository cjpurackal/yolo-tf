from PIL import Image
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import cv2


img = '7e436e5bb7f3b1cd.jpg'


# Neual net predicted output box and class
prediction = [0,0.432529,0.540797,0.6434319999999999,0.510436]
def generate_predicted_image(prediction,img):

	fig,ax=plt.subplots(1)
	# output from neural net
	class_val = prediction[0]
	x_norm = prediction[1]
	y_norm = prediction[2]
	w_norm = prediction[3]
	h_norm = prediction[4]

	# input image for prediction 
	im = Image.open(img)
	w_,h_ = im.size

	x_mid = x_norm*w_
	y_mid = y_norm*h_
	w = w_*w_norm
	h = h_*h_norm

	x1 = (x_mid-w/2)
	y1 = (y_mid-h/2)



	ax.imshow(im)
	rect=patches.Rectangle((x1,y1),w,h,linewidth=1,edgecolor='b',facecolor="none")

	ax.add_patch(rect)
	fig.savefig("prediction.png")

	
generate_predicted_image(prediction,img)