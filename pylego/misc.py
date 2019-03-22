import numpy as np
from PIL import Image


def print_gradient(name):
    '''To be used with .register_hook'''
    return lambda grad: print(name, grad.norm(p=2).item())


def add_argument(parser, flag, type=None, **kwargs):
    """Wrapper to add arguments to an argument parser. Fixes argparse's
    behavior with type=bool. For a bool flag 'test', this adds options '--test'
    which by default sets test to on, and additionally supports '--test true',
    '--test false' and so on. Finally, 'test' can also be turned off by simply
    specifying '--notest'.
    """
    def str2bool(v):
        return v.lower() in ('true', 't', '1')
    if flag.startswith('-'):
        raise ValueError('Flags should not have the preceeding - symbols, -- will be added automatically.')
    if type == bool:
        parser.add_argument('--' + flag, type=str2bool, nargs='?', const=True, **kwargs)
        parser.add_argument('--no' + flag, action='store_false', dest=flag)
    else:
        parser.add_argument('--' + flag, type=type, **kwargs)


def get_subclass(module, base_class):
    ret = []
    for name in dir(module):
        obj = getattr(module, name)
        try:
            if issubclass(obj, base_class) and obj is not base_class:
                ret.append(obj)
                if len(ret) > 1:
                    raise ValueError("Module " + module.__name__ + " has more than one class subclassing " +
                                     base_class.__name__)
        except TypeError:  # 'obj' is not a class
            pass
    if len(ret) == 0:
        raise ValueError("Module " + module.__name__ + " doesn't have a class subclassing " + base_class.__name__)
    return ret[0]


def nan_check(debug_str, variable):
    print(debug_str, np.any(np.isnan(variable)))


def scaled_int(tensor, scale=1.0):
    return (tensor * float(scale)).astype(np.int32)


def save_comparison_grid(fname, *args, border_width=2, desired_aspect=1.0, format='nchw'):
    """Arrange image batches in a grid such that corresponding images in *args are next to each other.
    All images should be in range [0,1]."""
    assert np.all(args[0].shape == arg.shape for arg in args)
    args = np.array(args)
    if format == 'nchw':
        args = np.transpose(args, (0, 1, 3, 4, 2))
    else:
        assert format == 'nhwc'

    args = np.concatenate([args, np.zeros([args.shape[0], args.shape[1], border_width, args.shape[3], args.shape[4]])],
                          axis=2)
    args = np.concatenate([args, np.zeros([args.shape[0], args.shape[1], args.shape[2], border_width, args.shape[4]])],
                          axis=3)
    args = np.concatenate(args, axis=2)
    aspect_ratio = args.shape[2] / args.shape[1]
    scale_aspect = aspect_ratio / desired_aspect

    # we want to divide width by scale_aspect, or multiply height by it
    # want nH * nW = N, with nH / nW = S => nH = S * nW
    # nW = sqrt(N/S), nH = S*nW
    nW = np.sqrt(args.shape[0] / scale_aspect)
    nH = scale_aspect * nW

    w_aspect = (np.ceil(nW) * args.shape[2]) / (np.floor(nH) * args.shape[1])
    h_aspect = (np.floor(nW) * args.shape[2]) / (np.ceil(nH) * args.shape[1])
    wh_aspect = (np.ceil(nW) * args.shape[2]) / (np.ceil(nH) * args.shape[1])
    w_diff = (np.abs(w_aspect - desired_aspect), (np.floor(nH), np.ceil(nW)))
    h_diff = (np.abs(h_aspect - desired_aspect), (np.ceil(nH), np.floor(nW)))
    wh_diff = (np.abs(wh_aspect - desired_aspect), (np.ceil(nH), np.ceil(nW)))

    for _, (h, w) in sorted([w_diff, h_diff, wh_diff]):
        if h * w >= args.shape[0]:
            nH = h
            nW = w
            break
    nH, nW = int(nH), int(nW)

    if args.shape[1] <= args.shape[2]:  # keep space at bottom
        while True:
            args_block = args[:(nH - 1) * nW]
            if args_block.shape[0] < (nH - 1) * nW:
                nH -= 1
            else:
                break
        args_block = args_block.reshape(nH - 1, nW, args_block.shape[1], args_block.shape[2], args_block.shape[3])
        args_bottom = args[(nH - 1) * nW:]
        if args_bottom.shape[0] > 0:
            args_bottom = np.concatenate([args_bottom, np.zeros([args_block.shape[1] - args_bottom.shape[0],
                                                                 args_bottom.shape[1], args_bottom.shape[2],
                                                                 args_bottom.shape[3]])], axis=0)
            args_bottom = args_bottom[None, ...]
            args = np.concatenate([args_block, args_bottom], axis=0)
        else:
            args = args_block
    else:  # keep space at right
        while True:
            args_block = args[:nH * (nW - 1)]
            if args_block.shape[0] < nH * (nW - 1):
                nW -= 1
            else:
                break
        args_block = args_block.reshape(nH, nW - 1, args_block.shape[1], args_block.shape[2], args_block.shape[3])
        args_right = args[nH * (nW - 1):]
        if args_right.shape[0] > 0:
            args_right = np.concatenate([args_right, np.zeros([args_block.shape[0] - args_right.shape[0],
                                                               args_right.shape[1], args_right.shape[2],
                                                               args_right.shape[3]])], axis=0)
            args_right = args_right[:, None, ...]
            args = np.concatenate([args_block, args_right], axis=1)
        else:
            args = args_block

    args = np.transpose(args, (0, 2, 1, 3, 4))
    args = args.reshape(args.shape[0] * args.shape[1], args.shape[2] * args.shape[3], args.shape[4])
    args = np.concatenate([np.zeros([border_width, args.shape[1], args.shape[2]]), args], axis=0)
    args = np.concatenate([np.zeros([args.shape[0], border_width, args.shape[2]]), args], axis=1)

    if args.shape[-1] == 1:
        args = args[:, :, 0]

    im = Image.fromarray((args * 255).astype(np.uint8))
    im.save(fname)
