
import _init_paths

import logging
import os
import os.path as osp

import threading
import time

change_length_lock = threading.Lock()   # new user first
each_thread_lock = threading.Lock()

class get_detection_thread(threading.Thread):

    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name

    def run(self):
        while True:
            if change_length_lock.locked():
                continue
            print(self.name)
            time.sleep(1)

class tracker_length_control_thread(threading.Thread):

    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name

    def run(self):
        while True:
            if Not_find_newUser():
                continue
            try:
                change_length_lock.acquire()
                print(self.name)
                time.sleep(3)
            finally:            
                change_length_lock.release()
                time.sleep(5)

def Not_find_newUser():
    return True

c = tracker_length_control_thread("c").start()
a = get_detection_thread("a").start()
b = get_detection_thread("b").start()













