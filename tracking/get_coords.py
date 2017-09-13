import skvideo.io 
import cv2
import time
global fr
fr=0
reader=skvideo.io.FFmpegReader("/Users/timomueller/Desktop/trafficvid1.mp4")
for frame in reader.nextFrame():
	if fr > 50 and fr < 250:
		cv2.imwrite("pics/cars_" + str(fr) + ".png", frame)
	fr+=1



	
