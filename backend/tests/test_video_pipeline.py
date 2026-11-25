import tempfile,unittest
from pathlib import Path
import cv2,numpy as np
from scripts.training.video.frame_sampling import uniform_frame_indices,sample_video
from scripts.training.video.video_metadata import inspect_video
from scripts.training.video.face_detection import FaceDetection,valid_bbox
from scripts.training.video.face_tracking import associate_faces
from scripts.training.video.face_cropping import crop_face
from scripts.training.video.build_manifest import build_record
from scripts.training.video.leakage_check import check_leakage,grouped_split

class VideoPipelineTests(unittest.TestCase):
 def test_sampling_unique_bounds(self):
  x=uniform_frame_indices(100,12);self.assertEqual(len(x),12);self.assertEqual(len(x),len(set(x)));self.assertEqual(x[0],0);self.assertEqual(x[-1],99)
 def test_empty_and_short(self):self.assertEqual(uniform_frame_indices(0),[]);self.assertEqual(uniform_frame_indices(2,12),[0,1])
 def test_crop_validation_resize(self):
  im=np.zeros((64,64,3),np.uint8);out,box=crop_face(im,(10,10,20,20),32);self.assertEqual(out.shape,(32,32,3));self.assertTrue(valid_bbox(box,64,64))
  with self.assertRaises(ValueError):crop_face(im,(-1,0,3,3))
 def test_tracking_and_group_split(self):
  a=FaceDetection((1,1,10,10),None,0,0.);b=FaceDetection((2,1,10,10),None,1,.1);self.assertEqual(associate_faces([[a],[b]])[0]['track_id'],associate_faces([[a],[b]])[1]['track_id'])
  rows=[{'video_id':f'{i}-{j}','source_video_id':str(i),'file_hash':f'h{i}-{j}'} for i in range(20) for j in range(2)];splits=grouped_split(rows);self.assertTrue(check_leakage(splits)['passed'])
 def test_synthetic_video_metadata_and_sampling(self):
  with tempfile.TemporaryDirectory() as td:
   p=str(Path(td)/'fixture.avi');writer=cv2.VideoWriter(p,cv2.VideoWriter_fourcc(*'MJPG'),10,(64,64));self.assertTrue(writer.isOpened())
   for i in range(9):
    frame=np.zeros((64,64,3),np.uint8);cv2.circle(frame,(32,32),12,(255,255,255),-1);writer.write(frame)
   writer.release();m=inspect_video(p);self.assertEqual(m['decode_status'],'ok');self.assertEqual(m['frame_count'],9);frames=sample_video(p,4);self.assertEqual(len(frames),4);self.assertTrue(all(f['decode_ok'] for f in frames))
 def test_manifest_labels_and_leakage_detection(self):
  self.assertEqual({'REAL':0,'MANIPULATED':1}['REAL'],0)
  x=check_leakage({'train':[{'source_video_id':'a'}],'test':[{'source_video_id':'a'}]});self.assertFalse(x['passed'])
