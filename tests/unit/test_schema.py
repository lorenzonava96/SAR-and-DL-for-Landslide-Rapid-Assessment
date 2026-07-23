from datetime import date
from pathlib import Path
import pytest
from app.schemas import PipelineRequest

def test_request_requires_one_input():
    with pytest.raises(ValueError):
        PipelineRequest('x','ASCENDING',date(2025,1,1),date(2025,1,1),Path('w'))
