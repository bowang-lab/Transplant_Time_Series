import sys
import ipdb

# install seqlearn from: https://github.com/larsmans/seqlearn
from seqlearn.hmm import MultinomialHMM
from seqlearn.evaluation import bio_f_score
from seqlearn.perceptron import StructuredPerceptron
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

'''
sys.path.append("/home/osvald/Projects/Diagnostics/UHN_oct/models/common")
from dataloader_original import Dataset, collate_fn
root = "/home/osvald/Projects/Diagnostics/UHN/data/IS/CV1/"
train_path = root + "n_train_tensors/"
valid_path = root + "n_valid_tensors/"
'''
def type_to_numbers5(type):
    if type == "5type1":
        return 1
    elif type == "5type2":
        return 2
    elif type == "5type3":
        return 3
    elif type == "5type4":
        return 4
    elif type == "5type5":
        return 5


def type_to_numbers1(type):
    if type == "1type1":
        return 1
    elif type == "1type2":
        return 2
    elif type == "1type3":
        return 3
    elif type == "1type4":
        return 4
    elif type == "1type5":
        return 5


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




if __name__ == "__main__":

    """"""
    indices = list(range(2613))
    path = train_path
    data = Dataset(indices, path)
    loader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=collate_fn)
    # go over each patient and build the train data
    train_df = pd.DataFrame()
    train_seqlength = []
    train_y = pd.DataFrame()
    for batch, labels, seq_len in loader:
        data = batch[0].cpu().numpy()
        y = labels[0].cpu().numpy()
        train_df = pd.concat((train_df, pd.DataFrame(data=data)), axis=0)
        train_y = pd.concat((train_y, pd.DataFrame(data=y)), axis=0)
        train_seqlength.append(int(seq_len.cpu().numpy()[0]))

    print(train_df.shape)
    print(train_y.shape)

    train_y.columns = [
        "5type1",
        "5type2",
        "5type3",
        "5type4",
        "5type5",
        "1type1",
        "1type2",
        "1type3",
        "1type4",
        "1type5",
    ]

    tr_label_5y = train_y[train_y.columns[0:5]]
    tr_label_1y = train_y[train_y.columns[5:10]]
    tr_label_5y["label"] = tr_label_5y.idxmax(1)
    tr_label_1y["label"] = tr_label_1y.idxmax(1)

    tr_label_5y["label_numbers"] = tr_label_5y["label"].apply(type_to_numbers5)
    tr_label_1y["label_numbers"] = tr_label_1y["label"].apply(type_to_numbers1)

    indices = list(range(656))
    path = valid_path
    data = Dataset(indices, path)
    loader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_df = pd.DataFrame()
    test_seqlength = []
    test_y = pd.DataFrame()
    for batch, labels, seq_len in loader:
        data = batch[0].cpu().numpy()
        y = labels[0].cpu().numpy()
        test_df = pd.concat((test_df, pd.DataFrame(data=data)), axis=0)
        test_y = pd.concat((test_y, pd.DataFrame(data=y)), axis=0)
        test_seqlength.append(int(seq_len.cpu().numpy()[0]))

    print(test_df.shape)
    print(test_y.shape)

    test_y.columns = [
        "5type1",
        "5type2",
        "5type3",
        "5type4",
        "5type5",
        "1type1",
        "1type2",
        "1type3",
        "1type4",
        "1type5",
    ]

    te_label_5y = test_y[test_y.columns[0:5]]
    te_label_1y = test_y[test_y.columns[5:10]]
    te_label_5y["label"] = te_label_5y.idxmax(1)
    te_label_1y["label"] = te_label_1y.idxmax(1)

    te_label_5y["label_numbers"] = te_label_5y["label"].apply(type_to_numbers5)
    te_label_1y["label_numbers"] = te_label_1y["label"].apply(type_to_numbers1)

    """
    alldata = [np.array([])] * 63
    for i in range(0, 63):
        col = test_df.iloc[:, i].values
        col_uni = np.unique(col[~np.isnan(col)])
        alldata[i] = np.concatenate((alldata[i], col_uni), axis=0)
    ipdb.set_trace()
    np.savetxt("test_random.txt", alldata)
    assert 0
    """

    '''
        for 5 year prediction
    '''
    # hmm = MultinomialHMM()
    hmm = StructuredPerceptron()
    hmm.fit(train_df, tr_label_5y["label_numbers"], train_seqlength)
    pred = hmm.predict(test_df, test_seqlength)
    # print(roc_auc_score(te_label_5y["label_numbers"], pred, average="macro"))


    onehot_encoder = OneHotEncoder(sparse=False)
    pred = pred.reshape(len(pred), 1)
    pred = onehot_encoder.fit_transform(pred)
    label = te_label_5y["label_numbers"].values.reshape(
        len(te_label_5y["label_numbers"]), 1
    )
    label = onehot_encoder.fit_transform(label)

    auc_per_class = np.zeros((5,))
    for i in range(5):
        pred_temp = pred[:, i]
        label_temp = label[:, i]
        score = roc_auc_score(label_temp, pred_temp)
        auc_per_class[i] = score
    print(auc_per_class)
    print(np.mean(auc_per_class[:5]))

    '''
        for 1 year prediction
    '''
    hmm = StructuredPerceptron()
    hmm.fit(train_df, tr_label_1y["label_numbers"], train_seqlength)
    pred = hmm.predict(test_df, test_seqlength)

    onehot_encoder = OneHotEncoder(sparse=False)
    pred = pred.reshape(len(pred), 1)
    pred = onehot_encoder.fit_transform(pred)
    label = te_label_1y["label_numbers"].values.reshape(
        len(te_label_1y["label_numbers"]), 1
    )
    label = onehot_encoder.fit_transform(label)

    auc_per_class = np.zeros((5,))
    for i in range(5):
        pred_temp = pred[:, i]
        label_temp = label[:, i]
        score = roc_auc_score(label_temp, pred_temp)
        auc_per_class[i] = score
    print(auc_per_class)
    print(np.mean(auc_per_class[:5]))
