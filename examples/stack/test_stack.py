import time

import torch

from pylego.ops import Stack


BATCH_SIZE = 5
STACK_SIZE = 6
SEQ_SIZE = 100
IN_SHAPE = ()
INDICED = True


if __name__ == '__main__':
    torch.manual_seed(42)
    start = time.time()
    stack = Stack(BATCH_SIZE, STACK_SIZE, IN_SHAPE)
    stream = torch.randn(BATCH_SIZE, SEQ_SIZE, *IN_SHAPE, requires_grad=True)
    for t in range(SEQ_SIZE):
        elem = stream[:, t]
        print(t, 'elem:', elem)
        push_ind = torch.randint(2, (BATCH_SIZE,), dtype=torch.bool) & ~stack.full()
        if INDICED:
            push_ind = torch.nonzero(push_ind, as_tuple=True)[0]
        print('push_ind:', push_ind)
        stack.push(push_ind, elem[push_ind])
        pop_ind = torch.randint(2, (BATCH_SIZE,), dtype=torch.bool) & ~stack.empty()
        if INDICED:
            pop_ind = torch.nonzero(pop_ind, as_tuple=True)[0]
        print('pop_ind:', pop_ind)
        popped = stack.pop(pop_ind, True)
        print('popped:', popped)
        print('popped size:', popped.size())
        print('-------')

    non_empty = ~stack.empty()
    print('Non-empty:', torch.nonzero(non_empty, as_tuple=True)[0])
    stack.top(non_empty).sum().backward()
    print('Stream gradient:', stream.grad)
    end = time.time()
    print('Total time:', end-start)
