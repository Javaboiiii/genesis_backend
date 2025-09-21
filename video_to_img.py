import ffmpeg 

import os 

dir = os.getcwd()

print(dir)

input_file_path = dir + "\\test_video.mp4" 

print(input_file_path)

(ffmpeg
    .input(input_file_path)
    .filter('fps', fps=1)
    .output('./frames/frame_%d.png')
    .run()
)