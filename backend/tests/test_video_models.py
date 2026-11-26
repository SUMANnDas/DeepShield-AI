import unittest
import torch
from scripts.training.video.models import (
 EfficientNetFrameClassifier, EfficientNetBiGRU,
 EfficientNetTransformerVideoClassifier, TemporalTransformer,
)
from scripts.training.video.metrics import binary_metrics

class VideoModelTests(unittest.TestCase):
 def test_frame_model_and_backward(self):
  model=EfficientNetFrameClassifier();x=torch.rand(2,3,64,64);y=model(x);self.assertEqual(y.shape,(2,2));y.mean().backward()
 def test_temporal_mask_and_shapes(self):
  model=EfficientNetBiGRU(hidden_dim=16,layers=1);model.eval();x=torch.rand(2,3,3,64,64);mask=torch.tensor([[1,1,1],[1,0,0]],dtype=torch.bool)
  with torch.no_grad():o=model(x,mask)
  self.assertEqual(o['logits'].shape,(2,2));self.assertEqual(o['frame_scores'].shape,(2,3));self.assertEqual(o['attention'][1,1:].sum().item(),0.)
 def test_transformer_mask_and_metrics(self):
  m=TemporalTransformer(dim=32,max_frames=4,layers=1,heads=4);o=m(torch.rand(2,4,32),torch.tensor([[1,1,1,1],[1,1,0,0]],dtype=torch.bool));self.assertEqual(o['logits'].shape,(2,2));self.assertEqual(binary_metrics([0,1],[.1,.9])['accuracy'],1.)

 def test_efficientnet_transformer_is_an_end_to_end_video_model(self):
  model=EfficientNetTransformerVideoClassifier(max_frames=2,layers=1,heads=4);x=torch.rand(1,2,3,64,64);mask=torch.tensor([[1,1]],dtype=torch.bool)
  output=model(x,mask);self.assertEqual(output['logits'].shape,(1,2));output['logits'].mean().backward()
