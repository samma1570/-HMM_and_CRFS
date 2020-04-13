import torch
import torch.utils.data as Data



def show_batch(loader):
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):
            print("steop:{}, batch_x:{}, batch_y:{}".format(step, batch_x, batch_y))
import torch.nn as nn
embeds = nn.Embedding(2, 5)

if __name__ == '__main__':
    BATCH_SIZE = 5
    x = torch.linspace(1, 10, 20)
    y = torch.linspace(10, 1, 20)
    print(x)
    print(y)
    torch_dataset = Data.TensorDataset(x, y)
    # print(torch_dataset)
    loader = Data.DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )
    show_batch(loader)