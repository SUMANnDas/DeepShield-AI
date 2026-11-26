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

