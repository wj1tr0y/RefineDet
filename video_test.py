import cv2
import argparse
import os
import shutil
import subprocess
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "run test and get all result")
    parser.add_argument("video", 
        help = "path of test video")
    parser.add_argument("--gpuid",
        help = "The gpu chosen to run the model.", required=True)

    args = parser.parse_args()
    assert len(args.gpuid) == 1, "You only need to choose one gpu. But {} gpus are chosen.".format(args.gpuid)

    video_name = args.video

    if not os.path.exists(video_name):
        print "{} doesn't exist.".format(video_name)

    frame_save_dir = '../dataset/test/videocap'
    if os.path.exists(frame_save_dir):
        shutil.rmtree(frame_save_dir)
        os.mkdir(frame_save_dir)
    else:
        os.mkdir(frame_save_dir)

    cap = cv2.VideoCapture(video_name)
    frame_count = 1
    success = True
    while(success):
        success, frame = cap.read()
        cv2.imwrite(os.path.join(frame_save_dir, 'frame{}.jpg'.format(frame_count)), frame)
        frame_count += 1

    cap.release()

    cmd = "python run_test.py --gpuid {} --out-dir result --test-set videocap".format(args.gpuid)
    print(cmd)
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output = process.communicate()[0]
    print(output)