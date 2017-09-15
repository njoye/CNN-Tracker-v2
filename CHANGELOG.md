13/09/17
- Added emergency mode (setting threshold lower when score < 0, so that tracking snaps back on after a full occlusion)
- Added some more adaptability to the program
- Added more ideas in comments ...
- Implemented first (untested) darknet usage through shell
- Forgot to say that I added a gitignore for the video file

14/09/17
- Gitignore for the video file is now working, original video: https://www.youtube.com/watch?v=PNCJQkvALVc (for the full occlusion test, I added a black bar in the center to fake the occlusion)
- Added tracker.py as class, that runs one MDNet for tracking and gets fed with a frame and values :)
- Added variable length of result & result_bb in Tracker.py so that we're able to do a frame by frame analyzation
- Refactored the Tracker.py class with the code from run_tracker.py so that self.* contains class specific variables like "model"
- fixed bug where pytorch would try to use gpu even though opts['use_gpu'] equaled False
- Realized that my Laptop is way too slow for this kind of work
- Split model code into two functions and got it working so that frame by frame analyzation is able to work (seems slower though, have to double check)
- Figured that I need asynchronous callbacks for the Tracker.py class (namely: startTracking() and updateFrame()) so that the code doesn't get blocked by multiple trackers at once, also because then I can have a callback that feeds the next frame into each Tracker because they will definitely not run equally fast

15/09/17
- Figured that indentation is important in python (srsly, spent 20 minutes on figuring out why some functions didn't work)
- Async sucks (as it already did in JS, but at least I knew how to properly do it there)
