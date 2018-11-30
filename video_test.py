import cv2
import argparse
import os
import shutil
import subprocess
import zipfile
import time

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

    frame_save_dir = '../dataset/test/videocap'+ str(int(time.time()))
    if os.path.exists(frame_save_dir):
        shutil.rmtree(frame_save_dir)
        os.mkdir(frame_save_dir)
    else:
        os.mkdir(frame_save_dir)
    
    cap = cv2.VideoCapture(video_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_count = 1
    success = True
    while(success):
        success, frame = cap.read()
        if success:
            print 'Reading frames: {}\r'.format(frame_count),
            cv2.imwrite(os.path.join(frame_save_dir, 'frame{}.jpg'.format(frame_count)), frame)
            frame_count += 1
        else:
            print ''
    cap.release()

    if os.path.exists('result'):
        shutil.rmtree('result')
        os.mkdir('result')

    print 'Detecting pedestrian.....'
    cmd = "python run_test.py --gpuid {} --out-dir result --test-set {}".format(args.gpuid, frame_save_dir.split('/')[-1])
    print(cmd)
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output = process.communicate()
    print(output)

    videoWriter = cv2.VideoWriter('result.avi', cv2.VideoWriter_fourcc('X','V','I','D'), fps, size)
    frame_name = os.listdir('result')
    frame_name = sorted(frame_name, key=lambda x: int(x[5:-9]))
    for i in frame_name:
        frame = cv2.imread('result/' + i)
        videoWriter.write(frame)
    videoWriter.release()


    shutil.rmtree(frame_save_dir)