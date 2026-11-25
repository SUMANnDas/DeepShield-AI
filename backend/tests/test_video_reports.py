import unittest
from scripts.training.video.report import forensic_report
from backend.services.video_detection.schemas import VideoDetectionResponse

class VideoReportTests(unittest.TestCase):
 def test_report_schema_and_uncertainty(self):
  x=forensic_report('fixture',.8,[{'frame_index':2,'timestamp_seconds':.2,'score':.9}],{'decode_status':'ok'},'synthetic',{})
  self.assertEqual(x['prediction'],'MANIPULATED');self.assertFalse(x['certainty_claim']);self.assertTrue(x['frame_evidence'][0]['suspicious']);VideoDetectionResponse(video_id='x',prediction='REAL',probability_manipulated=.2,frame_evidence=[],interpretation='indicator')
