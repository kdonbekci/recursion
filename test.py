from utils import *
from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

train_experiments = set(['HEPG2-01', 'HEPG2-03', 'HEPG2-05', 'HEPG2-06', 'HEPG2-07'])
test_experiments = set(['HEPG2-02', 'HEPG2-04'])

train_df = pd.read_csv(os.path.join(RECURSION_TRAIN_DIR, 'train.csv'))
train_df['cell_line'] = [v[0] for v in train_df.id_code.str.split('-')]
train_df = train_df[train_df['cell_line'] == 'HEPG2']
train_dfs = []
for site in range(2):
    cp = train_df.copy()
    cp['site'] = site+1
    train_dfs.append(cp)
train_df = pd.concat(train_dfs)
test_df = train_df.query('experiment in @test_experiments')
test_df = test_df.reset_index(drop=True)
train_df = train_df.query('experiment in @train_experiments')
train_df = train_df.sample(frac=1).reset_index(drop=True) #SHUFFLE

class RxRx1(Dataset):
    def __init__(self, df, is_train_dataset=True):
        self.df = df
        self.is_train_dataset = is_train_dataset
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        exp, well, plate, site = row.experiment, row.well, row.plate, row.site
        img_channels = [read_image(get_image_path(exp, plate, well, site, ch, train=True)) for ch in range(1,7)]
        img = np.stack(img_channels, axis=2).T
        if self.is_train_dataset:
            return img, np.array([row.sirna.astype('int32')])
        return img

train_set = RxRx1(train_df)
test_set = RxRx1(test_df, is_train_dataset=False)
batch_size = 12
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=10)

num_epochs = 20
total_step = len(train_loader)
num_channels = 6
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1108)
trained_kernel = model._conv_stem.weight
new_conv = nn.Sequential(nn.Conv2d(num_channels, 32, kernel_size=(3,3), stride=(2,2), bias=False),
            nn.ZeroPad2d(padding=(0, 1, 0, 1)))
with torch.no_grad():
    new_conv[0].weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)
model._conv_stem = new_conv
model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(num_epochs):
    for i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target[:,0].long().cuda()
        #print(data.shape)
        outputs = model(data)
        loss = criterion(outputs, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                 .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    exit()
    torch.save(model.state_dict(), os.path.join(DUMP_DIR, f'baseline-{epoch+1}.pt'))