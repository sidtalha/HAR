import numpy as np
import matplotlib.pyplot as plt
import pickle


# %%
def proto(data):

    subsampling = 1
    X = data["feature"]

    test_id = [2, 4, 6, 8]
    train_id = [1, 3, 5, 7]


    lab = data["labels"]
    subjects = data["subject"]
    actions = np.unique(lab)
    k = 1
    for j in actions:
        h = lab == j
        lab[h] = k
        k += 1

    actions = np.unique(lab)


    train_p = np.isin(subjects, train_id)
    test_p = np.isin(subjects, test_id)

    action_p = np.isin(lab, actions)

    train_p = np.logical_and(train_p, action_p)
    test_p = np.logical_and(test_p, action_p)

    train_p = train_p.reshape(len(train_p))

    test_p = test_p.reshape(len(test_p))


    x_train = X[:, :, :, train_p]
    y_train = lab[train_p]
    x_test = X[:, :, :, test_p]
    y_test = lab[test_p]

# %% Normalization


    mn = np.mean(np.mean(np.mean(x_train, axis=3), axis=0), axis=0)

    st = np.std(np.std(np.std(x_train, axis=3), axis=0), axis=0)

    mn = mn.reshape((1, 1, x_train.shape[2], 1))
    st = st.reshape((1, 1, x_train.shape[2], 1))

    x_train = (x_train - mn) / st

    x_test = (x_test - mn) / st



    x_train = np.transpose(x_train, (3, 2, 0, 1))
    x_test = np.transpose(x_test, (3, 2, 0, 1))


    with open("save\\x_train", "wb") as f:
        pickle.dump(x_train.astype(float), f)

    with open("save\\y_train", "wb") as f:
        pickle.dump(y_train, f)

    with open("save\\x_test", "wb") as f:
            pickle.dump(x_test.astype(float), f)

    with open("save\\y_test", "wb") as f:
            pickle.dump(y_test, f)


    D = {"train_data": x_train, "train_label": y_train, "test_data": x_test, "test_label": y_test, "actions": actions}
    return D
