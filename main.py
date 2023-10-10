import PIL
import time
import torch
import torchvision
from Models import ViTResNet, BasicBlock
from Engine import train, evaluate

N_EPOCHS = 150
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 100

DATA_PATH = "/Dataset/data_batch_4"  # CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
transform = torchvision.transforms.Compose(
     [torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
     torchvision.transforms.RandomAffine(8, translate=(.15,.15)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


train_dataset = torchvision.datasets.CIFAR10(DATA_PATH, train=True,
                                        download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(DATA_PATH, train=False,
                                       download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN,
                                          shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST,
                                         shuffle=False)


model = ViTResNet(BasicBlock, [3, 3, 3])
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=.9,weight_decay=1e-4)
#lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[35,48],gamma = 0.1)

train_loss_history, test_loss_history = [], []
for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch)
    start_time = time.time()
    train(model, optimizer, train_loader, train_loss_history)
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
    evaluate(model, test_loader, test_loss_history)

print('Execution time')

OUT_PATH = "/Result/ViTRes.pt"
torch.save(model.state_dict(), OUT_PATH)