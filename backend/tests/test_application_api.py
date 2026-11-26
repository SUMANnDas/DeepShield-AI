import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import create_app
from backend.services.analysis_store import AnalysisStore
from backend.services.image_inference import ImageInferenceService
from scripts.training.video.storage_budget import VideoStorageBudget, directory_size


class ApplicationApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.app=create_app()
        self.app.state.analysis_store=AnalysisStore(Path(self.tmp.name)/'test.sqlite3')
        self.app.state.video_storage_budget=VideoStorageBudget(
            root=Path(self.tmp.name)/'video-data', minimum_free_bytes=0, project_root=None,
        )
        self.client=TestClient(self.app)

    def tearDown(self):
        self.client.close();self.tmp.cleanup()

    def test_models_and_system_endpoints(self):
        models=self.client.get('/api/v1/models').json()['items']
        self.assertEqual(len(models),3);self.assertTrue(all(x['status']=='available' for x in models))
        status=self.client.get('/api/v1/system').json();self.assertIn('cuda_available',status);self.assertIn('gpu_name',status);self.assertIn('cpu_name',status);self.assertIn('database_status',status)

    def test_image_validation_real_format_and_size(self):
        image=Image.new('RGB',(48,32),(80,120,160));stream=io.BytesIO();image.save(stream,format='PNG')
        decoded,metadata=ImageInferenceService.validate_image(stream.getvalue(),'sample.png','image/png')
        self.assertEqual(decoded.size,(48,32));self.assertEqual(metadata['format'],'PNG')
        with self.assertRaises(ValueError):ImageInferenceService.validate_image(b'not an image','bad.png','image/png')
        with self.assertRaises(ValueError):ImageInferenceService.validate_image(stream.getvalue(),'sample.jpg','image/png')

    def test_upload_to_history_and_json_report(self):
        stream=io.BytesIO();Image.new('RGB',(32,32),(90,90,90)).save(stream,format='PNG')
        result={'predicted_label':'REAL','real_probability':.75,'fake_probability':.25,'confidence_score':.75,'confidence_calibrated':False,'model_identifier':'fixture','inference_duration_ms':1.,'device':'cpu','preprocessing':{'model_input':'32x32'}}
        with patch('backend.app.api.routes.application.image_service.predict',return_value=result):
            response=self.client.post('/api/v1/analysis/images',data={'model_id':'resnet50'},files={'file':('sample.png',stream.getvalue(),'image/png')})
        self.assertEqual(response.status_code,200,response.text);item=response.json();self.assertEqual(item['result']['predicted_label'],'REAL')
        self.assertEqual(self.client.get('/api/v1/history').json()['total'],1)
        report=self.client.get(f"/api/v1/reports/{item['id']}.json");self.assertEqual(report.status_code,200)
        self.assertEqual(report.json()['sha256'],item['sha256'])
        printable=self.client.get(f"/api/v1/reports/{item['id']}.html")
        self.assertEqual(printable.status_code,200); self.assertIn('Print / Save as PDF',printable.text)

    def test_video_upload_is_metadata_only_and_history_persists(self):
        response=self.client.post('/api/v1/analysis/videos',files={'file':('fixture.mp4',b'bad','video/mp4')})
        self.assertEqual(response.status_code,415)
        self.assertEqual(self.client.get('/api/v1/history').json()['total'],0)
        self.assertEqual(directory_size(self.app.state.video_storage_budget.root),0)

    def test_video_upload_returns_prediction_when_validated_model_registered(self):
        import cv2
        import numpy as np
        video_path=Path(self.tmp.name)/'clip.avi'
        writer=cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,64))
        self.assertTrue(writer.isOpened())
        for _ in range(4):
            frame=np.zeros((64,64,3),dtype=np.uint8)
            cv2.circle(frame,(32,32),12,(200,200,200),-1)
            writer.write(frame)
        writer.release()
        fake_models=[{'id':'ffpp-run-1','name':'EfficientNet-B0 + BiGRU','architecture':'efficientnet-b0-bigru-attention','status':'available'}]
        fake_detect={'status':'ok','model_id':'ffpp-run-1','model_identifier':'efficientnet-b0-bigru-attention:ffpp-run-1:epoch-4','device':'cpu','inference_duration_ms':12.3,'report':{'prediction':'MANIPULATED','probability_manipulated':0.88,'frame_evidence':[{'frame_index':0,'timestamp_seconds':0.0,'probability_manipulated':0.88,'suspicious':True}],'preprocessing':{'size':224}}}
        with patch('backend.app.api.routes.application.list_video_models',return_value=fake_models), patch('backend.app.api.routes.application.get_video_model',return_value=fake_models[0]), patch('backend.app.api.routes.application.video_service.detect',return_value=fake_detect):
            response=self.client.post('/api/v1/analysis/videos',data={'model_id':'ffpp-run-1'},files={'file':('clip.avi',video_path.read_bytes(),'video/x-msvideo')})
        self.assertEqual(response.status_code,200,response.text)
        body=response.json()
        self.assertTrue(body['prediction_available'])
        self.assertEqual(body['status'],'complete')
        self.assertEqual(body['result']['prediction'],'MANIPULATED')
        self.assertEqual(directory_size(self.app.state.video_storage_budget.root),0)

