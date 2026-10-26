"""Local nearest-centroid hypothesis classifier. Labels describe loss types, never causes."""
from pathlib import Path
import json, random, math
LABELS=['availability','speed','quality']
ROOT=Path(__file__).parent
PATH=(ROOT/'data' if (ROOT/'data').is_dir() else ROOT)/'model_training.json'
def make_training():
    rng=random.Random(915); train=[]
    for label in LABELS:
        for _ in range(40):
            v=[rng.uniform(.04,.18) for _ in LABELS]; v[LABELS.index(label)]=rng.uniform(.6,.9); z=sum(v)
            train.append({'features':[x/z for x in v],'label':label})
    PATH.write_text(json.dumps({'synthetic':True,'seed':915,'feature_order':LABELS,'records':train},indent=2),encoding='utf-8')
def classify(features):
    records=json.loads(PATH.read_text(encoding='utf-8'))['records']
    centroids={label:[sum(r['features'][i] for r in records if r['label']==label)/40 for i in range(3)] for label in LABELS}
    distances=sorted((math.sqrt(sum((x-y)**2 for x,y in zip(features,c))),label) for label,c in centroids.items())
    d,label=distances[0]
    return {'label':label if d<=.5 and distances[1][0]-d>=.08 else 'uncertain','distance':round(d,4),'model':'centroid-v1','confidence':'distance only, not calibrated probability'}
if __name__=='__main__':
    PATH.parent.mkdir(exist_ok=True); make_training()
