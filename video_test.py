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

    frame_save_dir = '../dataset/test/videoframe-'+ video_name[:video_name.index('.')]
    if not os.path.exists(frame_save_dir):
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


    out_dir = 'result' + str(int(time.time()))
    os.mkdir(out_dir)

    print 'Detecting pedestrian.....'
    cmd = "python run_test.py --gpuid {} --out-dir {} --test-set {}".format(args.gpuid, out_dir, frame_save_dir.split('/')[-1])
    print(cmd)
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output = process.communicate()

    print('Detection done. Now render results to video file.')

    out_video_name = video_name[:video_name.index('.')] + out_dir + '.avi'
    videoWriter = cv2.VideoWriter(out_video_name, cv2.VideoWriter_fourcc('X','V','I','D'), fps, size)
    frame_name = os.listdir(out_dir)
    frame_name = sorted(frame_name, key=lambda x: int(x[5:-9]))
    for i in frame_name:
        frame = cv2.imread(os.path.join(out_dir, i))
        videoWriter.write(frame)
    videoWriter.release()


    shutil.rmtree(frame_save_dir)
    shutil.rmtree(out_dir)

    cmd = "ffmpeg -threads 12 -y -i {} -strict experimental {}".format(out_video_name, out_video_name[:-4]+'.mp4')
    print(cmd)
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output = process.communicate()
    print(output)

    print('Done. Result was stored in {}'.format(out_video_name[:-4]+'.mp4'))
