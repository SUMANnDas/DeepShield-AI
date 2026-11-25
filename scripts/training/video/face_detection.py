from __future__ import annotations
from dataclasses import dataclass
import cv2

@dataclass
class FaceDetection:
    bbox: tuple[int,int,int,int]
    confidence: float | None
    frame_index: int
    timestamp_seconds: float | None

class FaceDetector:
    def detect(self,rgb,frame_index:int,timestamp_seconds=None)->list[FaceDetection]: raise NotImplementedError

class HaarFaceDetector(FaceDetector):
    """Local OpenCV Haar fallback; weaker than modern learned detectors."""
    def __init__(self,scale_factor=1.1,min_neighbors=5,min_size=(24,24)):
        cascade=cv2.data.haarcascades+'haarcascade_frontalface_default.xml'
        self.classifier=cv2.CascadeClassifier(cascade)
        if self.classifier.empty(): raise RuntimeError('OpenCV Haar cascade unavailable')
        self.scale_factor,self.min_neighbors,self.min_size=scale_factor,min_neighbors,min_size
    def detect(self,rgb,frame_index:int,timestamp_seconds=None):
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY); boxes=self.classifier.detectMultiScale(gray,self.scale_factor,self.min_neighbors,minSize=self.min_size)
        return [FaceDetection(tuple(map(int,b)),None,frame_index,timestamp_seconds) for b in boxes]

def valid_bbox(box,width,height):
    x,y,w,h=map(int,box); return x>=0 and y>=0 and w>0 and h>0 and x<width and y<height and x+w<=width and y+h<=height
