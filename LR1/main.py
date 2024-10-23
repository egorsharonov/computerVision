import cv2
import numpy as np
import os


class object_recognizer():
  def __init__(self, video_path, result_path):
    self.mark_color = None
    self.video_path = video_path
    self.result_path = result_path


  def load_video(self):
    self.in_video = cv2.VideoCapture(self.video_path)
    if not self.in_video.isOpened(): raise Exception
    fps = self.in_video.get(cv2.CAP_PROP_FPS)
    resolution = (int(self.in_video.get(3)), int(self.in_video.get(4)))
    self.writer = cv2.VideoWriter(self.result_path,
                           cv2.VideoWriter_fourcc(*'mp4v'),
                           fps,
                           resolution)


  def anylize_frame(self, frame):
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_bound = (5,50,50)
    upper_bound = (21, 200, 200)
    frame_threshold = cv2.inRange(frame_HSV, lower_bound, upper_bound)

    contours, _ = cv2.findContours(frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) <= 0: return None
    largest_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest_contour)

    return None if moments["m00"] == 0 else (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))


  def draw(self, frame, pos):
    cv2.drawMarker(frame, pos, color=self.mark_color,
                     markerSize=self.mark_size,
                     thickness=self.mark_thickness,
                     markerType=cv2.MARKER_STAR,
                     line_type=cv2.LINE_AA)


  def process_video(self):
    ret, frame_bgr = self.in_video.read()
    self.mark_color = (255, 0, 255)
    self.mark_size = 12
    self.mark_thickness = 1
    markers = []
    cnt = 0
    while ret:
      cnt += 1
      pos = self.anylize_frame(frame_bgr)
      if pos is not None and cnt%2==0:
        markers += [pos]
      [self.draw(frame_bgr, pos) for pos in markers]
      self.writer.write(frame_bgr)
      ret, frame_bgr = self.in_video.read()
    
    self.writer.release()


  def recognize(self):
    if not os.path.exists(self.video_path): raise Exception
    self.load_video()
    self.process_video()

object_recognizer("input.mp4", "output.mp4").recognize()