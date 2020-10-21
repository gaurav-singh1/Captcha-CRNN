import torch
import numpy as np
import os
import glob

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

import config
import dataset

from model import CaptchModel
import engine
from pprint import  pprint

def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j,:]:
            k = k - 1
            if k == -1:
                temp.append(".")
            else:
                temp.append(encoder.inverse_transform([k])[0])
        tp = "".join(temp)
        cap_preds.append(temp)
    return cap_preds




def run_training():
    image_files = glob.glob(os.path.join(config.DATA_DIR, "*.png"))[:10]
    target_orig = [x.split('/')[-1][:-4] for x in image_files]
    targets = [[c for c in x] for x in target_orig]
    targets_flat = [c for clist in targets for c in clist]

    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(targets_flat)
    targets_enc = [lbl_enc.transform(x) for x in targets]
    # didn't get this
    targets_enc = np.array(targets_enc) + 1
    # print(targets_enc)
    # print(len(lbl_enc.classes_))

    train_imgs, test_imgs, train_targets, test_targets, train_orig_targets, test_orig_targets = model_selection.train_test_split(image_files, targets_enc, target_orig, test_size = 0.1, random_state = 42)

    train_dataset = dataset.ClassificationDataset(image_paths=train_imgs, targets=train_targets, resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True)

    test_dataset = dataset.ClassificationDataset(image_paths=test_imgs, targets=test_targets, resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=False) 

    model = CaptchModel(num_chars = len(lbl_enc.classes_))
    model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )

    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(model, train_loader, optimizer)
        valid_preds, valid_loss = engine.eval_fn(model, test_loader)
        valid_cap_preds = []
        for vp in valid_preds:
            current_preds = decode_predictions(vp, lbl_enc)
            valid_cap_preds.extend(current_preds)
        pprint(list(zip(test_orig_targets, valid_cap_preds)))
        print(f"Epoch = {epoch}, TrainLoss = {train_loss}, ValidLoss = {valid_loss}")


if __name__ == "__main__":
    run_training()
