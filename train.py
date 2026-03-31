import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import time
import cv2
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils import *
from torch.autograd import Variable
from IPython.display import clear_output
from MedSAM.UNetFormer_MMSAM import UNetFormer as MFNet
try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

net = MFNet(num_classes=N_CLASSES).cuda()

params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print('All Params:   ', params)

params1 = 0
params2 = 0
for name, param in net.image_encoder.named_parameters():
    # if "Adapter" not in name:
    if "lora_" not in name:
    # if "lora_" not in name and "Adapter" not in name:
        params1 += param.nelement()
    else:
        params2 += param.nelement()
print('ImgEncoder:   ', params1)
# print('Adapter:       ', params2)
print('Lora: ', params2)
# print('Adapter_Lora: ', params2)
print('Others: ', params-params1-params2)

# for name, parms in net.named_parameters():
#     print('%-50s' % name, '%-30s' % str(parms.shape), '%-10s' % str(parms.nelement()))

# params = 0
# for name, param in net.sam.prompt_encoder.named_parameters():
#     params += param.nelement()
# print('prompt_encoder: ', params)

# params = 0
# for name, param in net.sam.mask_decoder.named_parameters():
#     params += param.nelement()
# print('mask_decoder: ', params)

# print(net)

print("training : ", len(train_ids))
print("testing : ", len(test_ids))
train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)

base_lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
# We define the scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [12, 17, 22], gamma=0.1)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5, 7, 9], gamma=0.1)

def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    test_batch_size = 32
    
    if DATASET == 'Potsdam':
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32') for id in test_ids)
    elif DATASET == 'Vaihingen' and IS_PREPROCESSED:
        test_images = []
        test_dsms = []
        test_labels = []
        eroded_labels = []
        for id in test_ids:
            data_pattern = TEST_DATA_FOLDER.format(id)
            dsm_pattern = TEST_DSM_FOLDER.format(id)
            label_pattern = TEST_LABEL_FOLDER.format(id)
            
            data_files = sorted(glob(data_pattern))
            dsm_files = sorted(glob(dsm_pattern))
            label_files = sorted(glob(label_pattern))
            
            for df in data_files:
                img = np.array(Image.open(df), dtype='float32')
                test_images.append(1 / 255 * img)
            for sf in dsm_files:
                dsm = np.array(Image.open(sf), dtype='float32')
                dsm = (dsm - np.min(dsm)) / (np.max(dsm) - np.min(dsm) + 1e-8)
                test_dsms.append(dsm)
            for lf in label_files:
                label = np.array(Image.open(lf), dtype='uint8')
                label[label == 6] = 5
                test_labels.append(label)
                eroded_labels.append(label)
        
        test_count = len(test_images)
        test_images = iter(test_images)
        test_dsms = iter(test_dsms)
        test_labels = iter(test_labels)
        eroded_labels = iter(eroded_labels)
    else:
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
        test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids)
        test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
        if DATASET == 'Hunan':
            eroded_labels = ((np.asarray(io.imread(ERODED_FOLDER.format(id)), dtype='int64')) for id in test_ids)
        else:
            eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)

    all_preds = []
    all_gts = []

    test_count = len(test_ids)
    
    with torch.no_grad():
        for img, dsm, gt, gt_e in tqdm(zip(test_images, test_dsms, test_labels, eroded_labels), total=test_count, leave=False):
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))

            all_coords = list(sliding_window(img, step=stride, window_size=window_size))
            total_batches = (len(all_coords) + test_batch_size - 1) // test_batch_size
            
            dsm_min = np.min(dsm)
            dsm_max = np.max(dsm)
            if DATASET == 'Hunan':
                dsm_normalized = (dsm - dsm_min) / (dsm_max - dsm_min + 1e-8)
            else:
                dsm_normalized = (dsm - dsm_min) / (dsm_max - dsm_min)
            
            for batch_idx in tqdm(range(total_batches), leave=False):
                start_idx = batch_idx * test_batch_size
                end_idx = min(start_idx + test_batch_size, len(all_coords))
                batch_coords = all_coords[start_idx:end_idx]
                
                image_patches = np.array([img[x:x + w, y:y + h].transpose((2, 0, 1)) for x, y, w, h in batch_coords], dtype='float32')
                dsm_patches = np.array([dsm_normalized[x:x + w, y:y + h] for x, y, w, h in batch_coords], dtype='float32')
                
                image_patches = torch.from_numpy(image_patches).cuda()
                dsm_patches = torch.from_numpy(dsm_patches).cuda()
                
                outs = net(image_patches, dsm_patches, mode='Test')
                outs = outs.data.cpu().numpy()
                
                for out, (x, y, w, h) in zip(outs, batch_coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out

            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt_e)
            clear_output()
    
    if DATASET == 'Hunan':
        accuracy = metrics_loveda(np.concatenate([p.ravel() for p in all_preds]),
                        np.concatenate([p.ravel() for p in all_gts]).ravel())
    else:
        accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                        np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    iter_ = 0
    MIoU_best = 0.00
    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()
        start_time = time.time()
        for batch_idx, (data, dsm, target) in enumerate(train_loader):
            data, dsm, target = Variable(data.cuda()), Variable(dsm.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data, dsm, mode='Train')
            loss = loss_calc(output, target, weights)
            # loss = CrossEntropy2d(output, target, weight=weights)
            loss.backward()
            optimizer.step()

            losses[iter_] = loss.data
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:
                clear_output()
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data, accuracy(pred, gt)))
            iter_ += 1

            del (data, target, loss)

        if e % save_epoch == 0:
            train_time = time.time()
            print("Training time: {:.3f} seconds".format(train_time - start_time))
            # We validate with the largest possible stride for faster computing
            net.eval()
            MIoU = test(net, test_ids, all=False, stride=Stride_Size)
            net.train()
            test_time = time.time()
            print("Test time: {:.3f} seconds".format(test_time - train_time))
            if MIoU > MIoU_best:
                if DATASET == 'Vaihingen':
                    os.makedirs('./resultsv', exist_ok=True)
                    torch.save(net.state_dict(), './resultsv/{}_epoch{}_{}'.format(MODEL, e, MIoU))
                elif DATASET == 'Potsdam':
                    os.makedirs('./resultsp', exist_ok=True)
                    torch.save(net.state_dict(), './resultsp/{}_epoch{}_{}'.format(MODEL, e, MIoU))
                elif DATASET == 'Hunan':
                    os.makedirs('./resultsh', exist_ok=True)
                    torch.save(net.state_dict(), './resultsh/{}_epoch{}_{}'.format(MODEL, e, MIoU))
                MIoU_best = MIoU
    print('MIoU_best: ', MIoU_best)

if MODE == 'Train':
    train(net, optimizer, epochs, scheduler, weights=WEIGHTS, save_epoch=save_epoch)

elif MODE == 'Test':
    if DATASET == 'Vaihingen':
        net.load_state_dict(torch.load('./resultsv/YOUR_MODEL'), strict=False)
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./resultsv/inference_UNetFormer_{}_tile_{}.png'.format('huge', id_), img)

    elif DATASET == 'Potsdam':
        net.load_state_dict(torch.load('./resultsp/YOUR_MODEL'), strict=False)
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./resultsp/inference_UNetFormer_{}_tile_{}.png'.format('huge', id_), img)

    elif DATASET == 'Hunan':
        net.load_state_dict(torch.load('./resultsh/YOUR_MODEL'), strict=False)
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=128)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./resultsh/inference_UNetFormer_{}_tile_{}.png'.format('base', id_), img)