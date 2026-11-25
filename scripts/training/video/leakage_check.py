from __future__ import annotations
from collections import defaultdict

def check_leakage(splits:dict[str,list[dict]]):
    violations=[]; seen={}
    for split,records in splits.items():
        for r in records:
            for key in ('video_id','source_video_id','file_hash'):
                value=r.get(key)
                if value is None: continue
                if key=='source_video_id' and r.get('label_name')=='MANIPULATED':
                    # source grouping still checked for all records where supplied
                    pass
                token=(key,str(value))
                if token in seen and seen[token]!=split: violations.append({'field':key,'value':str(value),'splits':[seen[token],split]})
                else: seen[token]=split
    return {'passed':not violations,'violations':violations}

def grouped_split(records,ratios=(.8,.1,.1),seed=42):
    import random
    if abs(sum(ratios)-1)>1e-8:raise ValueError('ratios must sum to 1')
    groups=defaultdict(list)
    for r in records:groups[r['source_video_id']].append(r)
    keys=sorted(groups); random.Random(seed).shuffle(keys); n=len(keys); a=round(n*ratios[0]); b=round(n*(ratios[0]+ratios[1]))
    result={name:[r for g in subset for r in groups[g]] for name,subset in zip(('train','validation','test'),(keys[:a],keys[a:b],keys[b:]))}
    check=check_leakage(result)
    if not check['passed']:raise ValueError(f'leakage detected: {check}')
    return result
