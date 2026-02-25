import torch
import pathlib
from bpp.utils import gen_instance


def generate_validation_data(n=20):
    pathlib.Path('../data/bpp').mkdir(parents=True, exist_ok=True)

    torch.manual_seed(12345)
    inst_list = []
    for _ in range(10):  # Small validation set
        demands = gen_instance(n, 'cpu')
        inst_list.append(demands)
    valDataset = torch.stack(inst_list)
    torch.save(valDataset, f'../data/bpp/valDataset-{n}.pt')
    print(f"Generated validation dataset for N={n}")


if __name__ == "__main__":
    generate_validation_data()
