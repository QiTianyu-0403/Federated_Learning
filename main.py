import argparse
from train.train_resnet18 import train_resnet18
from train.train_cnn import train_cnn
from train.train_lstm import train_lstm
from train.train_mobilenet import train_mobilenet
import FL_models.FedAvg as FL
import FL_models.HierFL as HFL
import FL_models.HFL_drl as HFL_drl
import FL_models.FedAvg_ray as FL_ray
import FL_models.HFL_ray as HFL_ray

def main():
    """
    for example:
    python main.py -m resnet18 -d Cifar10 -bs 128 -e 200
    python main.py -m lstm -d Shakespeare -bs 128 -e 20
    """
    parser = argparse.ArgumentParser(description='Federated Learning')
    # parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
    # parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
    '''Dataset, model, batch size, epoch'''
    parser.add_argument("-m", "--model", help="resnet18 or lstm or cnn or mobilenet", type=str, default='cnn')
    parser.add_argument("-d", "--data", help="Cifar or  MINIST or FMNIST or Shakespeare", type=str, default='MNIST')
    parser.add_argument("-bs", "--batchsize", help="the batch size of each epoch", type=int, default=128)
    parser.add_argument("-e", "--EPOCH", help="the number of epochs", type=int, default=10)
    parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=0.01)
    parser.add_argument("-nm", "--noniid_model", help="quantity_noniid or label_noniid or iid", type=str, default='iid')
    parser.add_argument("-iu", "--idx_user", help="Select the client number(<=num_users)", type=int, default=0)
    parser.add_argument("-nu", "--num_users", help="The number of clients", type=int, default=2)
    parser.add_argument("-ts", "--total_samples", help="The total samples of each clients", type=int, default=25000)
    parser.add_argument("-ru", "--rate_unbalance", help="The proportion of noniid (<=1.0) ", type=float, default=0.6)
    parser.add_argument("-nc", "--num_class", help="The classes number of noniid (<=10) ", type=int, default=2)

    '''Federated Learning'''
    parser.add_argument("-fm", "--FL_model", help="the model of FL: FL/HFL", type=str,default="HFL_ray")
    parser.add_argument("-p", "--port", help="the port used for rpc initialization", type=str,default="29500")
    parser.add_argument("-a", "--addr", help="the addr used for server", type=str, default="192.168.0.105")
    parser.add_argument("-r", "--rank", help="rank of this process", type=int, default=0)
    parser.add_argument("-tn", "--topo_num", help="the num of the topo", type=list, default=[3, 2, 2])
    parser.add_argument("-ws", "--world_size", help="number of process in group", type=int, default=3) # world_size 
    parser.add_argument("-ew", "--epoch_worker", help="the num of per worker run", type=int, default=3) # epoch_worker
    parser.add_argument("-ee", "--epoch_edge", help="the num of per edge run", type=int, default=2) # epoch_edge
    
    '''DRL'''
    parser.add_argument("-tf", "--traj_fre", help="The frequency of trajector", type=int, default=5)
    parser.add_argument("-ea", "--epoch_agent", help="The epoch of agent to learn", type=int, default=10)
    parser.add_argument("-g", "--greedy", help="select greedy", type=float, default=0.01)
    args = parser.parse_args()

    """
    for single worker train (the traditional codes)
    """
    # if args.model == 'resnet18':
    #     train_resnet18(args)
    # if args.model == 'cnn':
    #     train_cnn(args)
    # if args.model == 'lstm':
    #     train_lstm(args)
    # if args.model == 'mobilenet':
    #     train_mobilenet(args)

    '''
    This part is for RPC training, including the following:
    FedAvg:
    1 server and k workers
    HierFedAvg:
    1 server and n clients and k workers
    '''
    if args.FL_model == 'FL':
        FL.run_worker(args)
    elif args.FL_model == 'HFL':
        HFL.run_worker(args)
    elif args.FL_model == 'HFL_drl':
        HFL_drl.run_worker(args)
    elif args.FL_model =='FL_ray':
        FL_ray.run_worker(args)
    elif args.FL_model == 'HFL_ray':
        HFL_ray.run_worker(args)


if __name__ == "__main__":
    main()
