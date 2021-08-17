import os

import _init_paths
from opts import opts
from tracking_utils.utils import mkdir_if_missing
import datasets.dataset.jde as datasets
from track import eval_seq

from cv2 import cv2

def recogniton():
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)
    print("start tracking")
    path = "rstp://aifoundry:Coieeb1(@140.134.208.212:554/chID=0&steamType=sub"
    dataloader = datasets.LoadVideo(0, opt.img_size)
    print(dataloader.__dict__)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else os.path.join(result_root, 'frame')
    eval_seq(opt, dataloader, 'mot', result_filename,
             save_dir=frame_dir, show_image=True, frame_rate=frame_rate)


def testsss():
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
    path = "rstp://aifoundry:Coieeb1(@140.134.208.212:554/chID=0&steamType=sub"
    cap = cv2.VideoCapture(0, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
        exit()
    while(True):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destoryAllWindows()


if __name__ == '__main__':

    #testsss()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    recogniton()

