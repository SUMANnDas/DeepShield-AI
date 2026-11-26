from pathlib import Path
import torch
from .storage_budget import StorageBudgetExceeded, VideoStorageBudget


class _BudgetedWriter:
    """Count Torch's streaming checkpoint writes against the shared video quota."""

    def __init__(self, handle, budget: VideoStorageBudget):
        self.handle = handle
        self.budget = budget
        self.used_bytes = budget.report().current_bytes
        self.quota_error = None

    def write(self, data):
        size = memoryview(data).nbytes
        try:
            self.budget.require_capacity(
                size,
                purpose="writing a video model checkpoint",
                current_bytes=self.used_bytes,
            )
        except StorageBudgetExceeded as error:
            self.quota_error = error
            raise
        try:
            written = self.handle.write(data)
        except BaseException:
            # A short or failed write may still have consumed disk space.
            self.used_bytes = self.budget.report().current_bytes
            raise
        self.used_bytes += written
        return written

    def flush(self):
        return self.handle.flush()

    def tell(self):
        return self.handle.tell()

    def seek(self, offset, whence=0):
        previous = self.handle.tell()
        position = self.handle.seek(offset, whence)
        if position < previous:
            raise OSError("budgeted checkpoint writer does not allow rewriting earlier bytes")
        return position

def save_checkpoint(path, model, optimizer=None, epoch=0, config=None, budget=None, extra=None):
    active_budget = budget or VideoStorageBudget()
    p = active_budget.validate_destination(Path(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        'epoch': epoch,
        'config': config,
    }
    if extra:
        payload.update(extra)
    with p.open('xb') as handle:
        writer = _BudgetedWriter(handle, active_budget)
        try:
            torch.save(payload, writer)
        except BaseException:
            if writer.quota_error is not None:
                raise writer.quota_error
            raise


def load_checkpoint(path, model=None, optimizer=None, map_location='cpu'):
    state = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(state['model'])
    if optimizer and state.get('optimizer'):
        optimizer.load_state_dict(state['optimizer'])
    return state
