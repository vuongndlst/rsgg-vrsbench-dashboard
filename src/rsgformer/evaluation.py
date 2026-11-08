"""PredCls VG metrics preserving class/IoU matching and explicit empty-prediction zero recall."""
from collections import defaultdict
import numpy as np
from .geometry import overlaps

class PredClsEvaluator:
    def __init__(self,num_predicates=50,ks=(20,50,100),iou=0.5):
        self.num_predicates=num_predicates;self.ks=ks;self.iou=iou
        self.recalls={c:{k:[] for k in ks} for c in ['gc','ngc']}
        self.per_pred={c:{k:defaultdict(list) for k in ks} for c in ['gc','ngc']};self.images=0
    def _hits(self,sample,pred_rels):
        gt=sample.relations;result=[];boxes=sample.boxes;labels=sample.labels
        iou=overlaps(boxes,boxes)
        for s,o,p in pred_rels:
            matches=(gt[:,2]==p)&(labels[gt[:,0]]==labels[s])&(labels[gt[:,1]]==labels[o])&(iou[gt[:,0],s]>=self.iou)&(iou[gt[:,1],o]>=self.iou)
            result.append(set(np.where(matches)[0].tolist()))
        return result
    def update(self,sample,pairs,scores):
        gt=sample.relations
        if len(gt)==0:return
        pairs=np.asarray(pairs,dtype=np.int64).reshape(-1,2);scores=np.asarray(scores)
        assert scores.shape==(len(pairs),self.num_predicates+1) and np.isfinite(scores).all()
        assert len(set(map(tuple,pairs)))==len(pairs),'duplicate_predicted_directed_pair'
        if len(pairs):
            assert pairs.min()>=0 and pairs.max()<len(sample.labels)
            assert np.all(pairs[:,0]!=pairs[:,1])
            fg=scores[:,1:];pred=1+fg.argmax(1);order=np.argsort(-fg.max(1),kind='stable')
            gc=np.column_stack([pairs[order],pred[order]])[:max(self.ks)]
            ng_order=np.argsort(-fg.ravel(),kind='stable')[:max(self.ks)];pi,pr=np.unravel_index(ng_order,fg.shape)
            ngc=np.column_stack([pairs[pi],pr+1])
        else:gc=ngc=np.empty((0,3),dtype=np.int64)
        for constraint,pred_rels in [('gc',gc),('ngc',ngc)]:
            hits=self._hits(sample,pred_rels)
            for k in self.ks:
                match=set().union(*hits[:k]) if hits else set()
                self.recalls[constraint][k].append(len(match)/len(gt))
                for predicate in np.unique(gt[:,2]):
                    mask=set(np.where(gt[:,2]==predicate)[0].tolist())
                    self.per_pred[constraint][k][int(predicate)].append(len(mask&match)/len(mask))
        self.images+=1
    def result(self):
        out={'images_evaluated':self.images,'aggregation':'image_mean_R; image_then_predicate_mean_mR','empty_prediction_policy':'zero_recall','tie_policy':'stable_candidate_order; parity tests avoid score ties'}
        for c in ['gc','ngc']:
            out[c]={}
            for k in self.ks:
                out[c]['R@'+str(k)]=float(np.mean(self.recalls[c][k])) if self.images else None
                values=[float(np.mean(self.per_pred[c][k][p])) if self.per_pred[c][k][p] else 0. for p in range(1,self.num_predicates+1)]
                out[c]['mR@'+str(k)]=float(np.mean(values)) if self.images else None
                out[c]['per_predicate_R@'+str(k)]=values
        return out
