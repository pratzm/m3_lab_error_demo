
import torch
import argparse
import sys
import numpy as np


class MultinomialLogReg(torch.nn.Module):
    def __init__(self, W, b):
        super(MultinomialLogReg, self).__init__()
        self.weights = torch.nn.Parameter(W)
        self.bias = torch.nn.Parameter(b)

    def forward(self, x):
        mult = torch.matmul(x, self.weights)
        add = torch.add(mult.T, self.bias).T
        return add

def parse_all_args():
    # Parses commandline arguments

    parser = argparse.ArgumentParser()

    #parser.add_argument("D",help="The order of polynomial to fit (int)", type=int)
    parser.add_argument("C", help="The number of classes (int)", type=int)
    parser.add_argument("train_x",help="The training set input data (npz)")
    parser.add_argument("train_y",help="The training set target data (npz)")
    parser.add_argument("dev_x",help="The development set input data (npz)")
    parser.add_argument("dev_y",help="The development set target data (npz)")

    parser.add_argument("-lr",type=float,\
            help="The learning rate (float) [default: 0.1]",default=0.1)
    parser.add_argument("-epochs",type=int,\
            help="The number of training epochs (int) [default: 100]",\
            default=100)
    parser.add_argument("-lambda", dest="la", type=float,
            help="The regularization coefficient (float) [default: 0.0]",
            default=0.0)
    return parser.parse_args()

def train(model,train_x,train_y,dev_x,dev_y,args):
    # define our loss function
    criterion =  torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        # make prediction
        y_pred = model(train_x)
        #print(model.weights)

        # prints % accuracy
        train_accuracy = ((torch.argmax(y_pred, 1) == train_y).sum() / train_y.shape[0]).item()
        print(train_accuracy)

        # compute loss
        loss = criterion(y_pred, train_y) + args.la * torch.linalg.norm(model.weights, ord="fro")

        # take gradient step
        optimizer.zero_grad() # reset the gradient values
        loss.backward()       # compute the gradient values
        optimizer.step()      # apply gradients

        # eval on dev 
        dev_y_pred = model(dev_x)
        dev_accuracy = ((torch.argmax(dev_y_pred, 1) == dev_y).sum() / dev_y.shape[0]).item()
        
        print("train accuracy = %.3f, dev accuracy = %.3f" % (train_accuracy, dev_accuracy))

    #print(model.weights)

def main(argv):
    # parse arguments
    args = parse_all_args()

    # load data
    train_x = torch.from_numpy(np.load(args.train_x).astype(np.float32))
    train_y = torch.from_numpy(np.load(args.train_y).astype(np.long)).type(torch.LongTensor)#.to(0)
    dev_x   = torch.from_numpy(np.load(args.dev_x).astype(np.float32))
    dev_y   = torch.from_numpy(np.load(args.dev_y).astype(np.long)).type(torch.LongTensor)#.to(0)

    # PROF HUTCHINSON:
    # Changing between the commented and uncommented 'W' initialization will demo the error.
    # When W is filled with 0.000000001, has expected behavior (or so I think?)
    # When W is filled with 0.0, the first iteration will have the weights be 0, then NaN after.
    
    # Printing the weights at each step will demo. There is a comment on line 63, that when
    # uncommented will print weights at each step. I appreciate your help figuring this out.
    
    # cmd line call: ./m3_lab.py 10 mnist/train_{x,y}.npy mnist/dev_{x,y}.npy -lambda 0.0001 -lr 0.1

    W = torch.full((train_x.shape[1], args.C), 0.000001)
    #W = torch.full((train_x.shape[1], args.C), 0.0)
    b = torch.zeros(args.C, 1)

    model = MultinomialLogReg(W, b)
    train(model,train_x,train_y,dev_x,dev_y,args)

if __name__ == "__main__":
    main(sys.argv)
