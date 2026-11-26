from __future__ import annotations

import torch
from .preprocess_video import preprocess_video
from .report import forensic_report


@torch.no_grad()
def predict_video(path,model,device='cpu',max_frames=12,size=224,detector=None,max_frame_pixels=16_000_000):
    prepared=preprocess_video(path,detector,max_frames,size=size,max_frame_pixels=max_frame_pixels)
    if prepared.get('error'):
        return {'status':'error','error':prepared['error'],'metadata':prepared['metadata']}
    if not prepared['frames']:
        return {
            'status': 'inconclusive',
            'reason': 'no_reliable_face_track',
            'metadata': prepared['metadata'],
        }

    sequence = torch.zeros(1, max_frames, 3, size, size, dtype=torch.float32)
    mask = torch.zeros(1, max_frames, dtype=torch.bool)
    count = min(max_frames, len(prepared['frames']))
    for index, item in enumerate(prepared['frames'][:count]):
        sequence[0, index] = torch.from_numpy(item['crop'].transpose(2, 0, 1).copy()).float() / 255.0
        mask[0, index] = True

    model.eval()
    if hasattr(model, 'encode_frames'):
        output = model(sequence.to(device), mask.to(device))
        probability = torch.softmax(output['logits'], dim=1)[0, 1].item()
        frame_scores = output['frame_scores'][0, :count].detach().cpu().tolist()
    else:
        logits = model(sequence[0, :count].to(device))
        frame_probabilities = torch.softmax(logits, dim=1)[:, 1]
        probability = frame_probabilities.mean().item()
        frame_scores = frame_probabilities.detach().cpu().tolist()

    evidence = [
        {
            'frame_index': item['frame_index'],
            'timestamp_seconds': item['timestamp_seconds'],
            'score': score,
        }
        for item, score in zip(prepared['frames'][:count], frame_scores)
    ]
    return {
        'status': 'ok',
        'report': forensic_report(
            prepared['metadata'].get('file_hash'),
            probability,
            evidence,
            prepared['metadata'],
            type(model).__name__,
            {
                'size': size,
                'sampling': 'uniform',
                'max_frames': max_frames,
                'face_detector': prepared['face_detector'],
            },
            prepared.get('errors'),
        ),
    }
