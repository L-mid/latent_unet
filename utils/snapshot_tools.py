
import torch
import os
import matplotlib.pyplot as plt

SNAPHSHOT_DIR = "tests/snapshots/"


def save_difference_plot(tensor1, tensor2, name="diff_plot.png"):
    diff = (tensor1 - tensor2).flatten().detach().cpu().numpy()

    plt.figure(figsize=(6, 4))
    plt.hist(diff, bins=100)
    plt.title("Snapshot Difference Histogram")
    plt.xlabel("Difference")
    plt.ylabel("Frequency")

    os.makedirs("tests/debug_plots", exist_ok=True)
    plot_path = os.path.join("tests/debug_plots", name)
    plt.savefig(plot_path)
    plt.close()
    print(f"[Snapshot Debug] Difference plot saved to {plot_path}")
    



def assert_tensor_close_to_snapshot(tensor, snapshot_name, kind, atol=7.0):
    path = os.path.join(SNAPHSHOT_DIR, snapshot_name)
    
    os.makedirs(SNAPHSHOT_DIR, exist_ok=True)

    if not os.path.exists(path):
        torch.save(tensor, path)
        raise AssertionError(f"Snapshot {snapshot_name} created. Rerun test.")
    
    expected = torch.load(path)
    print("Max diff:", (tensor - expected).abs().max().item())
    print("Mean diff:", (tensor - expected).abs().mean().item())

    if not torch.allclose(tensor, expected, atol=atol):
        save_difference_plot(tensor, expected, name=f"diff_{kind}.png")

        raise AssertionError(f"{snapshot_name} output differs from snapshot.")
    

