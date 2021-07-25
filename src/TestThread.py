from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq

import threading
import time
lock1 = threading.Lock()
lock2 = threading.Lock()
class MovieThread(threading.Thread):

    def __init__(self, opt):
        print("Thread is creating")
        threading.Thread.__init__(self)
        self.opt = opt
        self.run()

    #lock
    def lock(self):
        self.lock_layer1()
        self.lock_layer2()
    def lock_layer1(self):
        lock1.acquire()
        lock2.acquire()
        time.sleep(0.01)
        lock2.release()
        lock1.release()
    def lock_layer2(self):
        lock2.acquire()
        lock1.acquire()
        lock1.release()
        lock2.release()

    def run(self):
        print("thread is runing")
        demo(self.opt)

def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    dataloader = datasets.MergeVideo(opt.input_video, opt.input_video)
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    eval_seq(opt, dataloader,  'mot', result_filename,
             save_dir=frame_dir, show_image=True, frame_rate=frame_rate,
             use_cuda=opt.gpus!=[-1])

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    t1 = MovieThread(opt)
