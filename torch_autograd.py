import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)
# None
print(x.grad_fn)

y = x + 2
print(y)

print(y.grad_fn)

z = y * y * 3
# 元素求均值
out = z.mean()

print(z, out)

# 使用雅克比行列式连接标量输出和每层中的向量
# 即v是l相对于y的梯度
# 且y是x的函数，J是y相对于x的梯度（雅克比矩阵）
# 那么J^T*v是l相对于x的梯度
out.backward()
print(x.grad)

