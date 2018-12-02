import os
import cv2 as cv

"""
The aim of this py program is to make the images 0=Angry, 1=Disgust, 2=Fear, 
3=Happy, 4=Sad, 5=Surprise, 6=Neutralin image set to 3 documents : Angry Happy Others
"""
def main():
	documents = os.getcwd()
	docs = os.listdir(documents)
	# make a dir to store the processed pictures
	if not (os.path.exists('Angry')):
		os.makedirs('Angry')
	if not (os.path.exists('Happy')):
        os.makedirs('Happy')
    if not (os.path.exists('Others')):
        os.makedirs('Others')
    happy_image_path = os.path.join(documents,'Happy')
    angry_image_path = os.path.join(documents,'Angry')
    others_image_path = os.path.join(documents,'Others')
    for index in ['0','1','2','3','4','5','6']:
        Data_path = os.join(documents,index)
        if ((index !='0') | (index!='3')):
	        list = os.listdir(Data_path)
	        for i in range(0,len(list)):
	        	path = os.path.join(trainData_path,list[i])
	        	if os.path.isfile(path):
		        	img = cv.imread(path)
		    	    cv.imwrite(others_image_path,img)
		    	    if i %1000 == 0:
		    	    	print("we have process %f images"%i)
        else if (index =='0'):
            list = os.listdir(Data_path)
	        for i in range(0,len(list)):
	        	path = os.path.join(trainData_path,list[i])
	        	if os.path.isfile(path):
		        	img = cv.imread(path)
		    	    cv.imwrite(angry_image_path,img)
		    	    if i %1000 == 0:
		    	    	print("we have process %f images"%i)
        else:
            list = os.listdir(Data_path)
	        for i in range(0,len(list)):
	        	path = os.path.join(trainData_path,list[i])
	        	if os.path.isfile(path):
		        	img = cv.imread(path)
		    	    cv.imwrite(happy_image_path,img)
		    	    if i %1000 == 0:
		    	    	print("we have process %f images"%i)

if __name__ == "__main__":
	main()