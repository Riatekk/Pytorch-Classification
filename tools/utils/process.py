import torch
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import wandb

def accuracy(proba_batch, label_batch):
    correct = 0
    batch_size = label_batch.size(0)
    preds = torch.argmax(proba_batch, dim=-1)
    for i, pred in enumerate(preds):
        if pred == label_batch[i]:
            correct += 1
    return correct / batch_size

def train_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()

    for batch, (data_batch, label_batch) in enumerate(train_loader):
        
        if 'cuda' in str(device):
            data_batch, label_batch = data_batch.to(device), label_batch.to(device)
        
        optimizer.zero_grad()
        pred_batch = model(data_batch)
        loss = loss_fn(pred_batch, label_batch)
        loss.backward()
        optimizer.step()
        
        acc = accuracy(pred_batch, label_batch)
        print(f'\r Training - Batch {batch+1}/{len(train_loader)}, train_loss: {loss:.4f}, train_acc: {acc:.4f}', end='')
        wandb.log({"train_loss": loss, "train_acc": acc})
    return loss.item(), acc

def validate(model, test_loader, loss_fn, device):
    model.eval()

    count,loss,acc = 0,0,0
    for batch, (data_batch, label_batch) in enumerate(test_loader):
        count += 1
        with torch.no_grad():
            
            if 'cuda' in str(device):
                data_batch, label_batch = data_batch.to(device), label_batch.to(device)

            pred_batch = model(data_batch)
            loss += loss_fn(pred_batch, label_batch)
            acc += accuracy(pred_batch, label_batch)
            print(f'\r Validate - Batch {batch+1}/{len(test_loader)}, val_loss: {loss/count:.4f}, val_acc: {acc/count:.4f}', end='')
            wandb.log({"val_loss": loss/count, "val_acc": acc/count})

    return loss.item() / len(test_loader), acc / len(test_loader)

def train(model, train_loader, test_loader, optimizer, loss_fn, use_gpu=False, epochs=10):
    res = { 'train_loss' : [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    #wandb.watch(model, log_freq=100)

    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}', end='\n')
    
        start_time = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
        print()
        val_loss, val_acc = validate(model, test_loader, loss_fn, device)

        res['train_loss'].append(train_loss)
        res['train_acc'].append(train_acc)
        res['val_loss'].append(val_loss)
        res['val_acc'].append(val_acc)
        
        #print(f"\n Epoch {epoch+1}/{epochs}, Train acc={train_acc:.3f}, Val acc={val_acc:.3f}, Train loss={train_loss:.3f}, Val loss={val_loss:.3f} - Time : {time.time() - start_time:.2f}s")

    return res

def plot_convolution(in_channels, filter, data_train,title=''):
    with torch.no_grad():
        c = nn.Conv2d(kernel_size=(3,3),out_channels=1,in_channels=in_channels)
        c.weight.copy_(filter)
        fig, ax = plt.subplots(2,6,figsize=(8,3))
        fig.suptitle(title,fontsize=16)
        for i in range(5):
            im = data_train[i][0]
            ax[0][i].imshow(im[0])
            ax[1][i].imshow(c(im.unsqueeze(0))[0][0])
            ax[0][i].axis('off')
            ax[1][i].axis('off')
        ax[0,5].imshow(filter)
        ax[0,5].axis('off')
        ax[1,5].axis('off')
        #plt.tight_layout()
        plt.show()

def display_dataset(dataset, n=10,classes=None):
    fig,ax = plt.subplots(1,n,figsize=(15,3))
    mn = min([dataset[i][0].min() for i in range(n)])
    mx = max([dataset[i][0].max() for i in range(n)])
    for i in range(n):
        ax[i].imshow(np.transpose((dataset[i][0]-mn)/(mx-mn),(1,2,0)))
        ax[i].axis('off')
        if classes:
            ax[i].set_title(classes[dataset[i][1]])