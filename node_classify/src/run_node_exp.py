import sys
import os
sys.path.append("../..")
# from definitions import ROOT_DIR
# sys.path.append(ROOT_DIR)
from node_classify.utils.dataset_utils import DataLoader, random_planetoid_splits2
from node_classify.utils.param_utils import *
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch_geometric.utils import homophily
import seaborn as sns

def RunExp(args, dataset, data,S, U, Net, percls_trn, val_lb):
    def train(model, optimizer, data):
        model.train()
        optimizer.zero_grad()
        out = model(data,S,U)[data.train_mask]
        nll = F.nll_loss(out, data.y[data.train_mask])
        loss = nll
        loss.backward()

        optimizer.step()
        del out

    def test(model, data):
        model.eval()
        logits, accs, losses, preds = model(data,S,U), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(model(data,S,U)[mask], data.y[mask])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses

    appnp_net = Net(dataset, args)
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    permute_masks = random_planetoid_splits2
    data = permute_masks(data, dataset.num_classes, percls_trn, val_lb,args.seed)

    model, data = appnp_net.to(device), data.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history, val_acc_history = [], []
    Gamma_0, Gamma_1 = [], []

    time_run = []

    for epoch in range(args.epochs):
        t_st = time.time()
        train(model, optimizer, data)
        time_epoch = time.time() - t_st  # each epoch train times
        time_run.append(time_epoch)
        train(model, optimizer, data)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            # best_model_wts = copy.deepcopy(model.state_dict())
            # if args.net == 'MFNN2':
            #     TEST1 = appnp_net.hgc[0].fW.clone()
            #     Alpha1 = TEST1.detach().cpu().numpy()
            #     TEST2 = appnp_net.hgc[1].fW.clone()
            #     Alpha2 = TEST2.detach().cpu().numpy()
            #     Gamma_0 = abs(Alpha1)
            #     Gamma_1 = abs(Alpha2)

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break
    return test_acc, best_val_acc, Gamma_0, Gamma_1,  time_run



#没有单位阵减 没有参数调
if __name__ == '__main__':
    args = parse_args()
    Net = get_net(args.net)

    SEEDS = [1941488137, 4198936517, 983997847, 4023022221, 4019585660, 2108550661, 1648766618, 629014539, 3212139042,
             2424918363]

    dataset, data ,S, U= DataLoader(args.dataset, args)


    RPMAX = args.RPMAX
    homo = homophily(data.edge_index, data.y)
    Gamma_0 = None
    alpha = args.alpha
    train_rate = args.train_rate
    val_rate = args.val_rate
    percls_trn = int(round(train_rate * len(data.y) / dataset.num_classes))
    val_lb = int(round(val_rate * len(data.y)))
    TrueLBrate = (percls_trn * dataset.num_classes + val_lb) / len(data.y)
    print('True Label rate: ', TrueLBrate)

    args.C = len(data.y.unique())
    args.Gamma = Gamma_0

    Results0 = []
    Result_test = []
    Result_val = []
    time_results = []
    for RP in tqdm(range(args.runs)):
        args.seed = SEEDS[RP]
        test_acc, best_val_acc, Gamma_0, Gamma_1, time_run = RunExp(args, dataset, data, S, U,Net, percls_trn, val_lb)
        time_results.append(time_run)
        Results0.append([test_acc, best_val_acc])
        Result_test.append(test_acc)
        Result_val.append(best_val_acc)

        print(f'test_acc:{test_acc:.4f}, best_val_acc:{best_val_acc:.4f}\n')

    run_sum = 0
    epochsss = 0
    for i in time_results:
        run_sum += sum(i)
        epochsss += len(i)

    print("each run avg_time:", run_sum / (args.runs), "s")
    print("each epoch avg_time:", 1000 * run_sum / epochsss, "ms")

    test_acc_mean, val_acc_mean = np.mean(Results0, axis=0) * 100
    test_acc_std = np.sqrt(np.var(Results0, axis=0)[0]) * 100

    values = np.asarray(Results0)[:, 0]
    uncertainty = np.max(
        np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000), 95) - values.mean()))

    print(f'{args.net} on dataset {args.dataset}, in {args.runs} repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.4f} \t test acc std = {test_acc_std:.4f} \t val acc mean = {val_acc_mean:.4f}')


    f = open("/home/luwei/MFNN/node_classify/res/{}_{}.txt".format(args.net,args.dataset), 'a')
    f.write("lr:{}, wd:{} ,hid:{}, drop:{}, dprate:{}, alpha:{},acc_test:{},std:{},time1:{},time2:{}".format(args.lr, args.weight_decay,
                                                                                         args.hidden, args.dropout,args.dprate,args.alpha,
                                                                                         test_acc_mean,
                                                                                         uncertainty * 100,
                                                                                         run_sum / (args.runs),
                                                                                         1000 * run_sum / epochsss))
    f.write("\n")
    f.close()

