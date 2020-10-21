import torch
import config
from tqdm import tqdm

def train_fn(model, data_loader, optimizer):
    model.train()
    tk0 = tqdm(data_loader, total=len(data_loader))
    fin_loss = 0
    for data in tk0:
        for k, v in data.items():
            data[k] = v.to(config.DEVICE)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        fin_loss+=loss.item()
    return fin_loss / len(data_loader)

def eval_fn(model, data_loader):
    model.eval()
    tk0 = tqdm(data_loader, total=len(data_loader))
    fin_loss = 0
    fin_preds = []
    with torch.no_grad():
        for data in tk0:
            for k, v in data.items():
                data[k] = v.to(config.DEVICE)

            batch_preds, loss = model(**data)
            fin_loss+=loss.item()
            fin_preds.append(batch_preds)
    return fin_preds, fin_loss / len(data_loader)


