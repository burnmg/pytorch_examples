from __future__ import print_function
import torch


'''
# 1.
x = torch.ones(2, 2, requires_grad = True)
print(x)

y = x + 2

print(y)
print(y.grad_fn)

z = y * y * 3
out = z.mean()

print(z, out)

out.backward()

print(x.grad)
'''

# 2.

x = torch.tensor([2.0,2], requires_grad = True)
y = x * x
y.backward(torch.tensor([0.5,1]))
print(y.grad)