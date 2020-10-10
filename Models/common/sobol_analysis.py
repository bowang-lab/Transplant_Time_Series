from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np
import torch

from RT.model import RT
from class_saliency import *
from dataloader import Dataset, collate_fn
import ipdb
import pandas as pd

"""
root = "/home/osvald/Projects/Diagnostics/UHN/data/IS/CV1/"
train_path = root + "n_train_tensors/"
valid_path = root + "n_valid_tensors/"
"""


class Args:
    dataset = "valid"
    disable_cuda = False


args = Args()

if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.DoubleTensor")
else:
    args.device = torch.device("cpu")
    torch.set_default_tensor_type("torch.DoubleTensor")


def get_boundary_value():
    # change accordingly based on your data dimension
    indices = list(range(2613))
    path = train_path
    data = Dataset(indices, path)
    loader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=collate_fn)

    each_max = np.array([]).reshape(0, 63)
    each_min = np.array([]).reshape(0, 63)
    for batch, labels, seq_len in loader:
        data = batch[0].cpu().numpy()
        arr_max = np.amax(data, axis=0).reshape(1, 63)
        arr_min = np.amin(data, axis=0).reshape(1, 63)
        each_max = np.concatenate((each_max, arr_max), axis=0)
        each_min = np.concatenate((each_min, arr_min), axis=0)

    train_all_max = np.amax(each_max, axis=0).reshape(1, 63)
    train_all_min = np.amin(each_min, axis=0).reshape(1, 63)

    indices = list(range(656))
    path = valid_path
    data = Dataset(indices, path)
    loader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=collate_fn)

    each_max = np.array([]).reshape(0, 63)
    each_min = np.array([]).reshape(0, 63)
    for batch, labels, seq_len in loader:
        data = batch[0].cpu().numpy()
        arr_max = np.amax(data, axis=0).reshape(1, 63)
        arr_min = np.amin(data, axis=0).reshape(1, 63)
        each_max = np.concatenate((each_max, arr_max), axis=0)
        each_min = np.concatenate((each_min, arr_min), axis=0)

    val_all_max = np.amax(each_max, axis=0).reshape(1, 63)
    val_all_min = np.amin(each_min, axis=0).reshape(1, 63)

    all_max = np.amax(
        np.concatenate((val_all_max, train_all_max), axis=0), axis=0
    ).reshape(1, 63)
    all_min = np.amax(
        np.concatenate((val_all_min, train_all_min), axis=0), axis=0
    ).reshape(1, 63)

    A = np.concatenate((all_min, all_max), axis=0)
    return A.transpose()


if __name__ == "__main__":
    # First, get the boundary value and name for each variable
    boundary = get_boundary_value()

    inputs = get_inputs()
    inputs.extend(
        ["Tacrolimus Level", "Cyclosporine Level", "Sirolimus Level", "HCC Recurrence"]
    )
    inputs = [inputs[i] for i in range(len(inputs)) if i != 16]  # get rid of index

    problem = {"num_vars": 63, "names": inputs, "bounds": boundary}

    # change the location accordingly
    folder = "/home/osvald/Projects/Diagnostics/UHN_oct/results/Transformer/CV1/SIG/dim32_heads2_levels2/lr0.000647_b1_0.846878_b2_0.973694_drop0.277799_l2_0.00336"
    model = RT(
        input_size=63,
        d_model=32,
        output_size=10,
        h=2,
        rnn_type="RNN",
        ksize=3,
        n=1,
        n_level=2,
        dropout=0,
    ).to(args.device)
    model.load_state_dict(torch.load(folder + "/best_auc_model"))
    model.eval()

    with torch.no_grad():

        x = saltelli.sample(problem, 40, calc_second_order=False)
        x = torch.from_numpy(x).to(args.device)
        x = torch.unsqueeze(x, 0)

        outputs = model(x)
        outputs = outputs.cpu().numpy()

        rank = np.array([]).reshape(63, 0)
        for i in range(0, outputs.shape[1]):
            Si = sobol.analyze(problem, outputs[:, i], calc_second_order=False)
            # let's only check for first-order
            rank = np.concatenate((rank, Si["S1"].reshape(63, 1)), axis=1)

        rank = np.absolute(rank)
        mean = np.mean(rank, axis=1)

        df = pd.DataFrame(index=inputs, data=mean, columns=["x"])

        df["rank"] = df["x"].rank(ascending=False)

        df.to_csv("sobol.csv")
