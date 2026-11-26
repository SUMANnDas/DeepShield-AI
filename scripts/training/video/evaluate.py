from __future__ import annotations

import torch
from .metrics import binary_metrics
from .train import _autocast, autocast_dtype


@torch.no_grad()
def evaluate(model,loader,device='cpu',mixed_precision='off'):
    model.eval()
    dtype = autocast_dtype(device, mixed_precision)
    labels = []
    probabilities = []
    rows = []
    skipped = []
    for batch in loader:
        frames = batch['frames'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['label'].tolist()
        video_ids = batch.get('video_id', [str(len(rows) + i) for i in range(len(targets))])
        if frames.ndim in (3, 5):
            valid_sequences = mask.any(dim=1)
            for index, valid in enumerate(valid_sequences.tolist()):
                if not valid:
                    skipped.append({'video_id': video_ids[index], 'reason': 'no_valid_face_sequence'})
            keep = valid_sequences.nonzero(as_tuple=False).flatten()
            if not len(keep):
                continue
            frames = frames[keep]
            mask = mask[keep]
            targets = [targets[index] for index in keep.tolist()]
            video_ids = [video_ids[index] for index in keep.tolist()]
        if frames.ndim == 5:
            if hasattr(model, 'encode_frames'):
                with _autocast(device, dtype):
                    output = model(frames, mask)
                batch_probabilities = torch.softmax(output['logits'].float(), dim=1)[:, 1].cpu().tolist()
                frame_scores = output.get('frame_scores')
            else:
                valid_frames = mask.reshape(-1)
                flat = frames.reshape(-1, *frames.shape[2:])[valid_frames]
                with _autocast(device, dtype):
                    logits = model(flat)
                frame_probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
                batch_probabilities = []
                offset = 0
                for row_mask in mask:
                    count = int(row_mask.sum())
                    batch_probabilities.append(frame_probabilities[offset:offset + count].mean().item())
                    offset += count
                frame_scores = None
        elif frames.ndim == 3:
            with _autocast(device, dtype):
                output = model(frames, mask)
            batch_probabilities = torch.softmax(output['logits'].float(), dim=1)[:, 1].cpu().tolist()
            frame_scores = output.get('frame_scores')
        else:
            with _autocast(device, dtype):
                output = {'logits': model(frames)}
            batch_probabilities = torch.softmax(output['logits'].float(), dim=1)[:, 1].cpu().tolist()
            frame_scores = None

        probabilities.extend(batch_probabilities)
        labels.extend(targets)
        for index, probability in enumerate(batch_probabilities):
            row = {
                'video_id': video_ids[index],
                'label': int(targets[index]),
                'probability_manipulated': float(probability),
            }
            if frame_scores is not None:
                row['frame_scores'] = frame_scores[index][mask[index]].detach().cpu().tolist()
            rows.append(row)

    if not labels:
        return {
            'status': 'no_valid_face_sequences',
            'metrics': None,
            'videos': [],
            'skipped': skipped,
        }
    return {
        'status': 'evaluated',
        'metrics': binary_metrics(labels, probabilities),
        'videos': rows,
        'skipped': skipped,
    }
