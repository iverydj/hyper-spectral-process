import matplotlib
import pytest
import torch

matplotlib.use("Agg")
torch.set_num_threads(4)


@pytest.fixture(autouse=True)
def close_figures():
    import matplotlib.pyplot as plt

    yield
    plt.close("all")
