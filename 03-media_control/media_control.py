import os
from argparse import ArgumentParser
import time
from collections import deque
from statistics import mode

from keras import models
from pynput import keyboard
import cv2
import numpy as np

class Application:

  GESTURES = ['like', 'no_gesture', 'dislike', 'stop']
  WIDTH = 640
  HEIGHT = 480
  IMG_SIZE = 64
  SIZE = (IMG_SIZE, IMG_SIZE)
  COLOR_CHANNELS = 3
  CURR_DIR = os.path.dirname(__file__)
  FPS = 1/5
  DQ_MAX_LEN = 5

  def __init__(self, device_id: int) -> None:
    self._load_model()
    self.device_id = device_id

    self._video_capture = cv2.VideoCapture(device_id)
    #force the webcam image to a smaller size to improve FPS
    #self._video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
    #self._video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)

    self.controller = keyboard.Controller()
    self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
    self.listener.start()

    self.running = True

    self.deque = deque([], maxlen=self.DQ_MAX_LEN)

  def _load_model(self) -> None:
    self.model = models.load_model(self.CURR_DIR + "/trained_model", compile=True)
    print("neural network loaded")

  def _process_image(self, img):
    if self.COLOR_CHANNELS == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img, self.SIZE)

    img_resized = np.array(img_resized).astype('float32')
    img_resized = img_resized / 255.
    img_resized = img_resized.reshape(-1, self.IMG_SIZE, self.IMG_SIZE, self.COLOR_CHANNELS)

    return img_resized
  
  def _on_press(self, key) -> None:
    pass

  def _on_release(self, key) -> None:
    if key == keyboard.Key.esc:
      self.running = False

  def _process_prediction(self, predicted_class) -> None:
    #like
    if predicted_class == 0:
      self.controller.press(key=keyboard.Key.media_volume_up)
      self.controller.release(key=keyboard.Key.media_volume_up)
    #no-gesture is 1

    #dislike
    elif predicted_class == 2:
      self.controller.press(key=keyboard.Key.media_volume_down)
      self.controller.release(key=keyboard.Key.media_volume_down)
    #stop
    elif predicted_class == 3:
      self.controller.press(key=keyboard.Key.media_play_pause)
      self.controller.release(key=keyboard.Key.media_play_pause)

  def run(self) -> None:
    while self.running:
      _, image = self._video_capture.read()
      processed_image = self._process_image(image)
      
      predicted_class = self.model.predict(processed_image, verbose='0').argmax(axis=-1)
      self.deque.append(predicted_class[0])

      if len(self.deque) == self.DQ_MAX_LEN:
        self._process_prediction(mode(self.deque))

      time.sleep(self.FPS)


if __name__ == "__main__":
  parser = ArgumentParser(prog="Media Controller", description="controlls a media player with pynput. like -> increase volume; dislike -> decrease volume; stop -> play/pause track")
  parser.add_argument("-d", default=0, type=int, help="id of webcam")

  args = parser.parse_args()

  application = Application(device_id=args.d)

  application.run()