import os
import multiprocessing as mp
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('thread_num', type=int)
parser.add_argument('--video_dir', type=str, default='datasets/activitynet/train_val_112')
parser.add_argument('--output_dir', type=str, default='datasets/activitynet/train_val_npy_112')
parser.add_argument('--max_frame_num', type=int, default=768)
args = parser.parse_args()

thread_num = args.thread_num
video_dir = args.video_dir
output_dir = args.output_dir
max_frame_num = args.max_frame_num
log_file=open('log.txt','w')


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


files = sorted(os.listdir(video_dir))

def sub_processor(pid, files):
    errorNO = 0
    npyedNO = 0
    for file in files[:]:
        # npyedNO=npyedNO+1
        # file_name = os.path.splitext(file)[0]
        # target_file = os.path.join(output_dir, file_name + '.npy')
        # cap = cv2.VideoCapture(os.path.join(video_dir, file))
        # count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # imgs = []
        # print(file,cap.isOpened())
        # while True:
        #     ret, frame = cap.read()
        #     if file_name=='v_--1DO2V4K74':
        #         pass
        #     else:
        #         if not ret:
        #             break
        #         imgs.append(frame[:, :, ::-1])
        # if len(imgs)==0:
        #     errorNO=errorNO+1
        #     log_file.write(file+'\n')
        #     log_file.write('【第一帧读取错误】\n\n')
        #     print('%d/%d %s frame num is less【第一帧读取错误】'%(errorNO,npyedNO,file_name))
        # if count != len(imgs):
        #     log_file.write(file + '\nframe num is less')
        #     print('{} frame num is less'.format(file_name))
        # if len(imgs) != 0:
        #     imgs = np.stack(imgs)
        #     print(file+str(imgs.shape))
        #     if max_frame_num is not None:
        #         imgs = imgs[:max_frame_num]
        #     np.save(target_file, imgs)
        file_name = os.path.splitext(file)[0]
        target_file = os.path.join(output_dir, file_name + '.npy')
        cap = cv2.VideoCapture(os.path.join(video_dir, file))
        #print(file, cap.isOpened())
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        imgs = []
        while True:
            ret, frame = cap.read()
            if not ret and len(imgs)!=0:
                break
            if not ret and len(imgs)==0:
                print('%s 【第一帧读取错误】' % (file_name))
                pass
            if ret:
                imgs.append(frame[:, :, ::-1])
        if count != len(imgs):
            print('------------------------------\n')
            print('--------------{} frame num is less----------------'.format(file_name))
            print('------------------------------\n')
        imgs = np.stack(imgs)
        print(file+str(imgs.shape))
        if max_frame_num is not None:
            imgs = imgs[:max_frame_num]
        np.save(target_file, imgs)


processes = []
video_num = len(files)
per_process_video_num = video_num // thread_num

for i in range(thread_num):
    if i == thread_num - 1:
        sub_files = files[i * per_process_video_num:]
    else:
        sub_files = files[i * per_process_video_num: (i + 1) * per_process_video_num]
    p = mp.Process(target=sub_processor, args=(i, sub_files))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

log_file.close()