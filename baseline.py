from utils import *
from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision

def build_model(num_classes=1108):
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
    # FIX FIRST CONV LAYER
    trained_kernel = model._conv_stem.weight
    new_conv = nn.Sequential(nn.Conv2d(6, 32, kernel_size=(3,3), stride=(2,2), bias=False),
                nn.ZeroPad2d(padding=(0, 1, 0, 1)))
    with torch.no_grad():
        new_conv[0].weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)
    model._conv_stem = new_conv
    return model

def predict(model, data_loader, return_truth = False):
    model.cuda()
    model.eval()
    predictions = []
    if return_truth:
        truth = []
    for i, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
        if return_truth:
            truth += list(target[:, 0].numpy())
        data, target = data.cuda(), target[:,0].long().cuda()
        with torch.no_grad():
            outputs = model(data)
        predictions+= list(outputs.argmax(dim=1).cpu().numpy())
    predictions = np.array(predictions).flatten()
    if return_truth:
        truth = np.array(truth).flatten()
        return predictions, truth
    return predictions

def train(model, train_loader, model_name, num_epochs, val_every,  
          save_every, timestamp=None, val_loader=None):
    
    if num_epochs is None:
        num_epochs = NUM_EPOCHS
    if val_every is None:
        val_every = VAL_EVERY
    if save_every is None:
        save_every = SAVE_EVERY
    history = {}
    if timestamp is None:
        timestamp = datetime.now().strftime("%m-%d_%H-%M")
    history['timestamp'] = timestamp
    history['loss'] = []
    history['predictions'] = {}
    history['accuracy'] = {}
    log_dir = os.path.join(LOG_DIR, '{}_{}'.format(model_name, timestamp))
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    model = model.cuda()
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    print('Beginning training for {} epochs.'.format(num_epochs))
    print('Will evaluate every {} epochs.'.format(val_every))
    print('Will checkpoint every {} epochs.'.format(save_every))
    for epoch in range(num_epochs):
        print(f'--------EPOCH {epoch+1}--------')
        for i, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.cuda(), target[:,0].long().cuda()
            outputs = model(data)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history['loss'].append(loss.detach().cpu().numpy())

            if (i+1) % (len(train_loader)//5) == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                     .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

        
        if (epoch + 1) % val_every == 0:
            if val_loader is not None:
                print('Evaluating model.')
                predictions, truth = predict(model, val_loader, return_truth=True)
                history['predictions'][epoch+1] = predictions
                accuracy = np.mean(predictions == truth) 
                history['accuracy'][epoch+1] = accuracy
                print('Val accuracy: {:.3f}'.format(accuracy))


        if (epoch + 1) % save_every == 0:
            print('Creating a checkpoint.')
            torch.save(model.state_dict(), os.path.join(log_dir, 
                                                        '{}_checkpoint_{:02}.pt'.format(model_name, epoch+1)))
            with open(os.path.join(log_dir, '{}_history_{:02}'.format(model_name, epoch+1)),'wb') as f:
                pickle.dump(history, f)
            
    return history

class RxRx1(Dataset):
    def __init__(self, df, is_train_dataset=True, augmentations=None, half_precision=False):
        self.df = df
        self.is_train_dataset = is_train_dataset
        self.augmentations = augmentations
        self.half_precision = half_precision
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        exp, well, plate, site = row.experiment, row.well, row.plate, row.site
        img_channels = [read_image(get_image_path(exp, plate, well, site, ch, train=True)) for ch in range(1,7)]
        img = np.stack(img_channels, axis=2).T
        if self.augmentations:
            for aug in self.augmentations:
                img = aug(img)
        if self.is_train_dataset:
            return tuple([img.astype(np.float16), np.array([row.sirna.astype('int16')])])  if self.half_precision else tuple([img, np.array([row.sirna.astype('int32')])])
        return img.astype(np.float16) if self.half_precision else img

def get_dataset_loaders(batch_size, num_workers, augmentations=None):
    train_df = pd.read_csv(os.path.join(RECURSION_TRAIN_DIR, 'train.csv'))
    train_df['cell_line'] = [v[0] for v in train_df.id_code.str.split('-')]
    train_df = train_df[train_df['cell_line'] == 'HEPG2']
    train_dfs = []
    for site in range(2):
        cp = train_df.copy()
        cp['site'] = site+1
        train_dfs.append(cp)
    train_df = pd.concat(train_dfs)
    test_df = train_df.query('experiment in @TEST_EXPERIMENTS')
    test_df = test_df.reset_index(drop=True)
    val_df  = train_df.query('experiment in @VAL_EXPERIMENTS')
    val_df = val_df.reset_index(drop=True)
    train_df = train_df.query('experiment in @TRAIN_EXPERIMENTS')
    train_df = train_df.sample(frac=1).reset_index(drop=True) #SHUFFLE
    train_set = RxRx1(train_df, augmentations=augmentations)
    test_set = RxRx1(test_df, augmentations=augmentations)
    val_set = RxRx1(val_df, augmentations=augmentations)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader