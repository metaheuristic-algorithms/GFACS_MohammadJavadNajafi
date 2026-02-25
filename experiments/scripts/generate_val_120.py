
import torch
import os
from bpp.utils import gen_instance


def generate_val_120():
    n = 120
    # Ensure directory exists
    os.makedirs('data/bpp', exist_ok=True)

    print(f"Generating validation dataset for N={n}...")
    torch.manual_seed(12345)
    inst_list = []
    # 10 instances for quick validation as per original N=20 setup being small?
    # Original utils.py does 100. Let's do 100 to be safe/standard.
    for _ in range(100):
        demands = gen_instance(n, 'cpu')
        inst_list.append(demands)
    valDataset = torch.stack(inst_list)
    torch.save(valDataset, f'data/bpp/valDataset-{n}.pt')
    print("Done.")


if __name__ == "__main__":
    generate_val_120()
