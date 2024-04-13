import numpy as np
import argparse

import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg,rotate
import pdb

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i_s', default=[100,300,600,97,225], help="index of the object to visualize")
#cls:[100,543,650,670,800,864]
    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output_seg')

    parser.add_argument('--exp_name', type=str, default="exp_seg", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device =  'cpu'

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object


    # ------ TO DO: Make Prediction ------
    num=0
    for a in [100,1000,5000]:
        print(a)
        # angle = a*np.pi/180
        # rotate_transform = rotate(angle)
        # test_data = rotate_transform.apply_rotation(test_data)
        ind = np.random.choice(10000,a, replace=False)
        test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
        test_label = torch.from_numpy((np.load(args.test_label))[:,ind])
        pred_label = model(test_data)
        pred_label = torch.argmax(pred_label,dim=-1)
        test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
        print ("test accuracy: {}".format(test_accuracy))
        # mask =pred_label.eq(test_label.data).cpu() == False
        # i_s = torch.where(mask[:,0])[0]

        # for i in i_s:
        #     viz_seg(test_data[i], test_label[i], "{}/fail_gt_{}_{}.gif".format(args.output_dir, args.exp_name,i), args.device)
        #     viz_seg(test_data[i], pred_label[i], "{}/fail_pred_{}_{}.gif".format(args.output_dir, args.exp_name,i), args.device)
        
        i = args.i_s[num]
        viz_seg(test_data[i], test_label[i], "{}/gt_{}_{}_{}.gif".format(args.output_dir, args.exp_name,a,i), args.device)
        viz_seg(test_data[i], pred_label[i], "{}/pred_{}_{}_{}.gif".format(args.output_dir,args.exp_name,a,i), args.device)
        pred_l = pred_label[i]
        test_l = test_label[i]
        test_accuracy = pred_l.eq(test_l.data).cpu().sum().item() / (test_l.reshape((-1,1)).size()[0])
        print ("accuracy_{}: {}".format(i,test_accuracy))
        num+=1


