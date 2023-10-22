import time

import torch.optim

import models.transform_layers as TL
from training.contrastive_loss import get_similarity_matrix, NT_xent
from utils.utils import AverageMeter, normalize

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)


def train(P, epoch, model, criterion, optimizer, scheduler, loader, logger=None,
          simclr_aug=None, linear=None, linear_optim=None):

    assert simclr_aug is not None
    assert P.sim_lambda == 1.0  # to avoid mistake
    assert P.K_shift > 1

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['sim'] = AverageMeter()
    losses['shift'] = AverageMeter()

    check = time.time()
    for n, (images, labels) in enumerate(loader):
        model.train()
        count = n * P.n_gpus  # number of trained samples

        data_time.update(time.time() - check)
        check = time.time()

        ### SimCLR loss ###
        # if dataset is NOT imagenet
        if P.dataset != 'imagenet':
            # assume batch_size == 4
            batch_size = images.size(0)
            images = images.to(device)
            # the horizontal flip is required by SimClr
            images1, images2 = hflip(images.repeat(2, 1, 1, 1)).chunk(2)  # hflip
            # each has four images
        else:
            batch_size = images[0].size(0)
            images1, images2 = images[0].to(device), images[1].to(device)
        labels = labels.to(device)

        # each has 16 images
        images1 = torch.cat([P.shift_trans(images1, k) for k in range(P.K_shift)])
        images2 = torch.cat([P.shift_trans(images2, k) for k in range(P.K_shift)])
        shift_labels = torch.cat([torch.ones_like(labels) * k for k in range(P.K_shift)], 0)  # B -> 4B
        shift_labels = shift_labels.repeat(2)
        
        # 32 images
        images_pair = torch.cat([images1, images2], dim=0)  # 8B
        images_pair = simclr_aug(images_pair)  
        # transform: color_jitter, color_gray, resize_crop 
        
        # TODO what is going on here?
        _, outputs_aux = model(images_pair, simclr=True, penultimate=True, shift=True)

        # outputs_aux['penultimate'] is of (length, -1)
        # outputs_aux['simclr'] is of (length, 128(simclr_dim))
        # outputs_aux['shift'] is of (length, 2), why 2?

        simclr = normalize(outputs_aux['simclr'])  # normalize
        sim_matrix = get_similarity_matrix(simclr, multi_gpu=P.multi_gpu)
        # Loss cls-SI
        loss_sim = NT_xent(sim_matrix, temperature=0.5) * P.sim_lambda
        # Loss con-SI
        loss_shift = criterion(outputs_aux['shift'], shift_labels)

        ### total loss ###
        loss = loss_sim + loss_shift

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step(epoch - 1 + n / len(loader))
        lr = optimizer.param_groups[0]['lr']

        batch_time.update(time.time() - check)

        ### Post-processing stuffs ###
        simclr_norm = outputs_aux['simclr'].norm(dim=1).mean()

        # what is going on here?
        # 0-3
        penul_1 = outputs_aux['penultimate'][:batch_size]
        # 16-19
        # both are unrotated images
        penul_2 = outputs_aux['penultimate'][P.K_shift * batch_size: (P.K_shift + 1) * batch_size]
        outputs_aux['penultimate'] = torch.cat([penul_1, penul_2])  # only use original rotation

        ### Linear evaluation ###
        outputs_linear_eval = linear(outputs_aux['penultimate'].detach())
        # the linear layer is judging whether A and flipped A are A
        loss_linear = criterion(outputs_linear_eval, labels.repeat(2))
        
        # what is linear?
        linear_optim.zero_grad()
        loss_linear.backward()
        linear_optim.step()

        losses['cls'].update(0, batch_size)
        losses['sim'].update(loss_sim.item(), batch_size)
        losses['shift'].update(loss_shift.item(), batch_size)

        if count % 50 == 0:
            log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossC %f] [LossSim %f] [LossShift %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['cls'].value, losses['sim'].value, losses['shift'].value))

    log_('[DONE] [Time %.3f] [Data %.3f] [LossC %f] [LossSim %f] [LossShift %f]' %
         (batch_time.average, data_time.average,
          losses['cls'].average, losses['sim'].average, losses['shift'].average))

    if logger is not None:
        logger.scalar_summary('train/loss_cls', losses['cls'].average, epoch)
        logger.scalar_summary('train/loss_sim', losses['sim'].average, epoch)
        logger.scalar_summary('train/loss_shift', losses['shift'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)

