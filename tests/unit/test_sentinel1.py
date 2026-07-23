import numpy as np
from app.preprocessing.sentinel1 import compose_stack

def test_compose_stack_band_order():
    pre_vv=np.ones((2,2)); pre_vh=np.full((2,2),2)
    post_vv=np.full((2,2),4); post_vh=np.full((2,2),8)
    result=compose_stack(pre_vv,pre_vh,post_vv,post_vh)
    assert result.shape==(2,2,4)
    assert np.all(result[...,0]==4); assert np.all(result[...,1]==8)
    assert np.all(result[...,2]==3); assert np.all(result[...,3]==6)
