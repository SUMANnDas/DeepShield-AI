from pathlib import Path
import torch

def save_checkpoint(path,model,optimizer=None,epoch=0,config=None):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict() if optimizer else None,'epoch':epoch,'config':config},f)
def load_checkpoint(path,model,optimizer=None,map_location='cpu'):
    state=torch.load(path,map_location=map_location,weights_only=False);model.load_state_dict(state['model'])
    if optimizer and state.get('optimizer'):optimizer.load_state_dict(state['optimizer'])
    return state
