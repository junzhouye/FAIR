from models.cifar10_resnet import *
from loss.similarity_loss import similarity_loss, constrastive_similarity_loss


def standard_train_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        optimizer.zero_grad()
        # calculate robust loss
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        # print progress
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def sim_standard_train(model, device, train_loader, optimizer, epoch):
    """
    训练模型精度上升慢，存在模型震荡的问题，模型的精度在一个个epoch中精度变化很大。
    精度在50%左右踌躇不前，loss变化可以很大。
    后期loss基本是在 -200 ~ -1000 之间了
    然后loss已经到了 -10,000的水准。这一项只可能是loss_sim带来的
    因为模型的输出经过了relu层，因此模型的feature 都是正值 而这时候 sim不可能是负数

    这个大失败。不行。
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    # max
    sim_loss = similarity_loss(device)
    # min
    con_sim_loss = constrastive_similarity_loss(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        feature, outputs = model(data, True)
        optimizer.zero_grad()
        # calculate robust loss
        loss_CE = criterion(outputs, target)
        loss_sim = sim_loss(feature, target)
        loss_con_sim = con_sim_loss(feature, target)
        loss = loss_CE + 0.1 * loss_con_sim - 0.1 * loss_sim
        loss.backward()
        optimizer.step()
        # print progress
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def sim_standard_train_V2(model, device, train_loader, optimizer, epoch):
    """
    V2:相比V1，去除了其中一个损失项。该损失，只关注不同类之间的相似度，而不对相同类的相似度给出要求。
    理性分析就是，相同类的类内也存在差异 将这种差异抹除 使它们相似 可能是很不适宜的。

    就目前的epoch来看，效果大大滴提升。模型精度也比较高。
    这个可以从损失函数的设计角度来看，本来的设计思路是通过一个batch中的feature的similarity来作为损失，但是我写的损失，没有严格按照公式，也就是
    cos-similarity的公式，这样就存在一个问题，它的similarity可以无限大
    但是不同类的损失就不一样，因为relu的限制，其loss限制在了[0,+infinity) 它没有上限但是有下限

    但是 相比正常训练 它的精度还是下降了 这是不难预见的.

    class-wise fair 对公平性没有什么帮助，在正常精度上仍然是精度具有差异，在对抗攻击上，相似类仍然是更容易互相转换

    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    # min
    con_sim_loss = constrastive_similarity_loss(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        feature, outputs = model(data, True)
        optimizer.zero_grad()
        # calculate robust loss
        loss_CE = criterion(outputs, target)
        loss_con_sim = con_sim_loss(feature, target)
        loss = loss_CE + 0.1 * loss_con_sim
        loss.backward()
        optimizer.step()
        # print progress
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def sim_standard_train_V3(model, device, train_loader, optimizer, epoch):
    """
    这次尝试仅仅针对两个类进行训练。这里选择 dog 和 cat ，也就是 3 和 5
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    # min
    con_sim_loss = constrastive_similarity_loss(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        feature, outputs = model(data, True)
        optimizer.zero_grad()
        # calculate robust loss
        loss_CE = criterion(outputs, target)
        loss_con_sim = con_sim_loss(feature, target)
        loss = loss_CE + 0.1 * loss_con_sim
        loss.backward()
        optimizer.step()
        # print progress
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
