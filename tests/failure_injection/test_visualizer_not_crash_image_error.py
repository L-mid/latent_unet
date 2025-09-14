
import pytest
from utils.visualizer import visualize_everything
from model.model_stub import Model

model = Model()

@pytest.mark.failure_injection
def test_visuallizer_error_swallowed_in_calling_code(fp):
    vis = visualize_everything(model)
    fp.once("visualizer.add_image", exc=ValueError("invalid image"))

    # In real code, wrap calls: try/except ValueError to warn not crash.
    """
    try: 
        vis.add_image("sample", b"...", 0)
    except ValueError:
        # Simulate calling code converting to warning
        pass
    """
    
        