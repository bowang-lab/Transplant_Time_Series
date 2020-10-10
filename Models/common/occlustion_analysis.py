import numpy as np
import torch

from RT.model import RT
from class_saliency import *
from dataloader import Dataset, collate_fn
import ipdb
import pandas as pd
from distribution_fit import Distribution

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


def get_unique_value():
    # change accordingly based on your data dimension
    alldata = [np.array([])] * 63

    indices = list(range(2613))
    path = train_path
    data = Dataset(indices, path)
    loader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=collate_fn)
    # go over each patient
    for batch, labels, seq_len in loader:
        data = batch[0].cpu().numpy()
        for i in range(0, 63):
            temp = np.unique(data[:, i])
            alldata[i] = np.append(alldata[i], temp)

    indices = list(range(656))
    path = valid_path
    data = Dataset(indices, path)
    loader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=collate_fn)
    for batch, labels, seq_len in loader:
        data = batch[0].cpu().numpy()
        for i in range(0, 63):
            temp = np.unique(data[:, i])
            alldata[i] = np.append(alldata[i], temp)

    for i in range(0, 63):
        alldata[i] = np.unique(alldata[i])

    return alldata


if __name__ == "__main__":
    alldata = get_unique_value()

    # for each variable, fit a distruibution for that
    var_distributions = []
    for i in range(0, 63):
        var_dis = Distribution()
        var_dis.Fit(alldata[i])
        var_distributions.append(var_dis)

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

    real = [np.array([])] * 10

    perturb_all = []
    for i in range(0, 63):
        perturb = [np.array([])] * 10
        perturb_all.append(perturb)

    num_followups = 50

    with torch.no_grad():
        indices = list(range(656))
        path = valid_path
        data = Dataset(indices, path)
        loader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=collate_fn)
        count = 0
        for batch, labels, seq_len in loader:
            print(count)
            batch, labels = batch.to(args.device), labels.to(args.device)
            seq_len = torch.clamp(seq_len, 1, num_followups)
            temp = batch[0][
                : int(seq_len[0]),
            ]
            batch = torch.unsqueeze(temp, 0)

            outputs = model(batch)
            outputs = outputs.cpu().numpy()

            for i in range(0, 10):
                real[i] = np.append(real[i], outputs[:, i])

            # Now lets perturb each variable and then run the model and store the output
            for i in range(0, 63):
                temp_batch = batch.clone().detach()
                for j in range(0, int(seq_len[0])):
                    temp_batch[0][j][i] = np.float64(var_distributions[i].Random()[0])

                outputs = model(temp_batch)
                outputs = outputs.cpu().numpy()
                for k in range(0, 10):
                    perturb_all[i][k] = np.append(perturb_all[i][k], outputs[:, k])
            count += 1

    from scipy.stats import entropy

    inputs = get_inputs()
    inputs.extend(
        ["Tacrolimus Level", "Cyclosporine Level", "Sirolimus Level", "HCC Recurrence"]
    )
    inputs = [inputs[i] for i in range(len(inputs)) if i != 16]  # get rid of index

    kldiv = []
    for i in range(0, 63):
        temp = 0
        for j in range(0, 10):
            temp += entropy(real[j], perturb_all[i][j])
        temp = temp / 10
        kldiv.append(temp)

    df = pd.DataFrame(index=inputs, data=kldiv, columns=["x"])
    df["rank"] = df["x"].rank(ascending=False)
    df.to_csv("occlusion.csv")
