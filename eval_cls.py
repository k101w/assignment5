import numpy as np
import argparse

import torch
from models import cls_model,cls_model_2
from utils import create_dir
import pdb
from utils import viz_cls,rotate
def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp_cls", help='The name of the experiment')
    parser.add_argument('--i_s', default=[100,543,650,670,800,864], help='The name of the experiment')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    # args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    args.device = 'cpu'
    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model_2(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls_2/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object

    # ------ TO DO: Make Prediction ------
    num=0
    for a in [5000]:
    # [30,45,60,90,150]:
        print(a)
        ind = np.random.choice(10000,a, replace=False)
        test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:]).to(args.device)
        test_label = torch.from_numpy(np.load(args.test_label))
        # angle = a*np.pi/180
        # rotate_transform = rotate(angle)
        # test_data = rotate_transform.apply_rotation(test_data)
        pred_label = model(test_data)
        pred_label = torch.argmax(pred_label,dim=-1)
        # for i in args.i_s:
        #     viz_cls(test_data[i], test_label[i], "{}/gt_{}_{}.gif".format(args.output_dir, args.exp_name,i), args.device)
        #     print(f'order:{i}')
        #     print(f'catergory:{pred_label[i]}')
        #     print(f'true catergory:{test_label[i]}')

        # Compute Accuracy
        test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
        # mask =pred_label.eq(test_label.data).cpu() == False
        # i_s = torch.where(mask)[0]
        for i in args.i_s:
            viz_cls(test_data[i], test_label[i], "{}/gt_{}_{}_{}.gif".format(args.output_dir, args.exp_name,a,i), args.device)
            print(f'order:{i}')
            print(f'catergory:{pred_label[i]}')
            print(f'true catergory:{test_label[i]}')


        print ("test accuracy: {}".format(test_accuracy))
        num+=1
        

