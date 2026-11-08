"""Lossless NumPy scalar/array conversion at JSON boundary, no scientific casting."""
import numpy as np
def numpy_json_default(value):
 if isinstance(value,np.generic):return value.item()
 if isinstance(value,np.ndarray):return value.tolist()
 raise TypeError('unsupported_JSON_object:'+type(value).__name__)
