
import torch

#generato

g =torch.Generator().manual_seed(2147483647)
p =torch.rand(3,generator=g)
print (p)
p=p/sum(p)
print (p)

print (
torch.multinomial(p, num_samples=20, replacement=True,generator=g))
torch.multinomial(p, num_samples=20, replacement=True,generator=g)