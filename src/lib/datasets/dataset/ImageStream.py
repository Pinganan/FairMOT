import functools
import os
import numpy as np
import cv2
import time
import threading
import queue

class Camera():
    camera = None

    def __init__(self, port):
        self.port = port
        self.connect()

    # @synchronized
    def read(self):
        return self.camera.read()

    # @synchronized
    def connect(self):
        global camera_url
        #print(self.port)
        camera_url = 'rtsp://aifoundry:Coieeb1(@140.134.208.{:d}:554/chID=0&streamType=main'.format(self.port)
        self.camera = cv2.VideoCapture(camera_url)
        if self.camera.isOpened():
            print('VideoCapture created')
        else:
            print('VideoCapture created fail')
        width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.camera.get(cv2.CAP_PROP_FPS)
        print('Resolution: ' + str(width) +' * ' + str(height))
        print('FPS: ' + str(fps))

    # @synchronized
    def reconnect(self):
        self.camera.release()
        self.connect()


class camera_Loader(threading.Thread):
    def __init__(self, queue, port, img_size=(1088, 608)):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True
        self.camera = Camera(port)

        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0
        self.w, self.h = 1920, 1080
        self.start()
    
    def run(self):
        self.looper()
        threading.Timer(0, self.run).start()

    def looper(self):
        self.count += 1
        # Read image
        ret, img0 = self.camera.read()
        if not ret:
            print('Disconnected. Trying to reconnect...')
            self.camera.reconnect()
            return
        img0 = cv2.resize(img0, (self.w, self.h))

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)
        
        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        # cv2.imwrite(path + '{:d}.jpg'.format(self.count), img)
        data = [self.count, img, img0]
        self.queue.put(data)

    def puttest(self):
        self.count += 1
        self.queue.put(self.count)
        print("producer: " + str(self.count))
        time.sleep(1)


class ImageStream():
    def __init__(self, port):
        self.buffer = queue.Queue()
        camera_Loader(self.buffer, port)

    def getImage(self):
        # get data
        data = self.buffer.get()
        # delete data
        for i in range(1):self.buffer.get()
        count, img, img0 = data
        return count, img, img0


def letterbox(img, height=608, width=1088,
            color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


if __name__ == "__main__":
    camera1 = ImageStream(225)
    camera2 = ImageStream(221)
    while(True):
        count, img, img0 = camera1.getImage()
        __, __, img02 = camera2.getImage()
        cv2.imshow("camera1", img0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow("camera2", img02)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   
    cv2.destroyAllWindows()
