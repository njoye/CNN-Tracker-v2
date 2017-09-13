import skvideo.io 
import cv2
from subprocess import call

interval=1
interval_counter=0

global fr
fr=0
reader=skvideo.io.FFmpegReader("/Users/timomueller/Downloads/wdstesting.mp4")
for frame in reader.nextFrame():
	if interval_counter == interval:
		name=str(fr)
		name=name.zfill(4)
		print("Created file: " + name + ".jpg")
		cv2.imwrite("img/" + name + ".jpg", frame)
		interval_counter=0
	else:
		interval_counter+=1
	fr+=1

call(["rm", ".DS_STORE"])