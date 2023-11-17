import torch

device = None
def decide_on_device(args):
    global device
    if args.use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')