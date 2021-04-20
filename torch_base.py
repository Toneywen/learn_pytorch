import torch 

if __name__ == "__main__":
    print(torch.__version__)

    x = torch.empty(5, 3)
    print(x)

    # 0 1 之间随机
    x = torch.rand(5, 3)
    print(x)

    x = torch.zeros(5, 3, dtype=torch.long)
    print(x)

    x = torch.tensor([5.5, 3])
    print(x)

    x = x.new_ones(5, 3, dtype=torch.double)      # new_* 方法来创建对象
    print(x)

    x = torch.randn_like(x, dtype=torch.float)    # 覆盖 dtype!
    print(x) 

    print(x.size())

    # 不同的加法
    y = torch.rand(5, 3)
    print(x + y)

    print(torch.add(x, y))

    result = torch.empty(5, 3)
    torch.add(x, y, out=result)
    print(result)

    y.add_(x)  # 这里会改变y的值
    print(y)

    # 表示拿到第1列的所有值
    print(x[:, 1])
    # 第0 列
    print(x[:, 0])
    # 第1行
    print(x[1, :])

    x = torch.randn(4, 4)
    y = x.view(16)
    print(y)
    z = x.view(-1, 8)  #  size -1 从其他维度推断
    print(x.size(), y.size(), z.size())

    x = torch.randn(1)
    print(x)
    # 获取元素的值（只能获取一个元素的值），即可以使用遍历达到获取所有元素的值
    print(x.item())

    a = torch.ones(5)
    print(a)

    b = a.numpy()
    print(b)

    # 原始值上主元素+1
    a.add_(1)
    print(a)
    print(b)

    import numpy as np
    a = np.ones(5)
    b = torch.from_numpy(a)
    np.add(a, 1, out=a)
    print(a)
    print(b)

    if torch.cuda.is_available():
        device = torch.device("cuda:1")          # a CUDA 设备对象
        y = torch.ones_like(x, device=device)  # 直接从GPU创建张量
        x = x.to(device)                       # 或者直接使用``.to("cuda")``将张量移动到cuda中
        z = x + y
        print(z)
        print(z.to("cpu", torch.double))       # ``.to`` 也会对变量的类型做更改