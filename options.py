import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--epoch', type=int, default=300, help="the number of epochs of original method: E")
    parser.add_argument('--local_bs', type=int, default=256, help="local batch size: B")
    parser.add_argument('--topk', type=int, default=-1, help="precision@topk")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--w_d', type=float, default=0.004, help='weight_decay')
    parser.add_argument('--lam', type=float, default=0.0001, help='CSQ参数')
    parser.add_argument('--alp', type=float, default=0.01, help='DSH参数')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
    parser.add_argument('--agg_method', type=str, default='fedavg', help='aggregation method')
    parser.add_argument('--algorithm', type=str, default='N', help='use N or not')
    parser.add_argument('--mu', type=float, default=0.1, help='The mu in fedprox which controls the weight of proximal term in loss function')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imgs")
    parser.add_argument('--alpha', type=float, default=1, help='kl loss')
    parser.add_argument('--beta', type=float, default=1, help='classify loss')
    parser.add_argument('--bit', type=int, default=16, help="Length of binary hash code")
    parser.add_argument('--hash_method', type=str, default='DSH', help="hash method")
    
    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--personal_method', type=str, default='N', help="juhefangfa")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_train', type=int, default=50000, help="number of train")
    parser.add_argument('--gpu', default=None, help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='adam', help="type of optimizer")
    parser.add_argument('--iid', type=int, default=1, help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0, help='whether to use unequal data splits for non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--partition', type=float, default=0.8, help='non-iid-partition')
    parser.add_argument('--epoch_lr_decrease', type=int, default=500, help='decrease of learning rate')
    parser.add_argument('--gamma', type=float, default=1, help='moon loss ')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--compute', type=int, default=1, help='duo shao lun ji suan yi ci')
    parser.add_argument('--K_pfedme', type=int, default=5, help='lcoal step')
    parser.add_argument('--lam_pfedme', type=float, default=15, help='chaocan pfedme')
    parser.add_argument('--lam_ditto', type=float, default=1, help='chaocan pfedme')
    args = parser.parse_args()
    return args
