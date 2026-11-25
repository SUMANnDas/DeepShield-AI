from __future__ import annotations
from datetime import datetime,timezone

def forensic_report(video_id,p_video,frame_scores,metadata,model_name,preprocessing,errors=None):
    return {'schema_version':'1.0','created_utc':datetime.now(timezone.utc).isoformat(),'video_id':str(video_id),'prediction':'MANIPULATED' if p_video>=.5 else 'REAL','probability_manipulated':float(p_video),'threshold':.5,'certainty_claim':False,'interpretation':'Model-based forensic indicator; not proof of authenticity or manipulation.','frame_evidence':[{'frame_index':int(x['frame_index']),'timestamp_seconds':x.get('timestamp_seconds'),'probability_manipulated':float(x['score']),'suspicious':bool(x['score']>=.5)} for x in frame_scores],'metadata':metadata,'model':model_name,'preprocessing':preprocessing,'errors':errors or []}
