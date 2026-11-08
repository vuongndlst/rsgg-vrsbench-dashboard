"""SAM native non-logit masks may be float0/1; reject arbitrary numeric logits."""
import numpy as np
def binary_mask(value):
 a=np.asarray(value)
 if a.ndim!=2:raise ValueError('SAM_mask_must_be_2D')
 if a.dtype==bool:return a
 if not np.isfinite(a).all() or not np.logical_or(a==0,a==1).all():raise ValueError('non_binary_SAM_mask_requires_explicit_logit_path')
 return a.astype(bool)
