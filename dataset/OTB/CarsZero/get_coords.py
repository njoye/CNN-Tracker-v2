import skvideo.io 
import cv2
from subprocess import call



global fr
fr=0
reader=skvideo.io.FFmpegReader("/Users/timomueller/Desktop/trafficvid1.mp4")
for frame in reader.nextFrame():
	if fr > 50 and fr < 250:
		name=str(fr)
		name=name.zfill(4)
		print(name)
		cv2.imwrite("img/" + name + ".jpg", frame)
	fr+=1

call(["rm", ".DS_STORE"])