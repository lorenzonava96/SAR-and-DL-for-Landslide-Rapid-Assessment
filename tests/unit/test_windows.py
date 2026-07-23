import numpy as np
from app.inference.windows import sliding_windows

def test_sliding_windows_includes_exact_edge():
    image=np.zeros((96,96,4),dtype=np.float32)
    windows=list(sliding_windows(image,size=64,step=32))
    assert len(windows)==4
    assert windows[-1][0].x1==96 and windows[-1][0].y1==96
