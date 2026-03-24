import argparse
import os
import random
import time

import numpy as np
import torch
from torch.cuda import amp

from Dataloader import dataloader
from Loss_fun import make_loss
from VID_Test import test
from VID_Trans_model import VID_Trans
from utility import AverageMeter, optimizer, scheduler


def set_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VID-Trans-ReID no-camera + supervised contrastive training')
    parser.add_argument('--Dataset_name', required=True, type=str)
    parser.add_argument('--model_path', required=True, type=str, help='ViT pretrained weight path')
    parser.add_argument('--output_dir', default='./output_camera_removed_supcon', type=str)
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--eval_every', default=10, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seq_len', default=4, type=int)
    parser.add_argument('--center_w', default=0.0005, type=float)
    parser.add_argument('--con_loss_w', default=0.01, type=float)
    parser.add_argument('--con_temp', default=0.07, type=float)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(1234)

    train_loader, _, num_classes, camera_num, view_num, q_val_loader, g_val_loader = dataloader(
        args.Dataset_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seq_len=args.seq_len
    )

    model = VID_Trans(
        num_classes=num_classes,
        camera_num=camera_num,
        pretrainpath=args.model_path
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    loss_fun, center_criterion = make_loss(
        num_classes=num_classes,
        contrast_temp=args.con_temp
    )
    center_criterion = center_criterion.to(device)

    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=0.5)
    optimizer_main = optimizer(model)
    lr_scheduler = scheduler(optimizer_main)
    scaler = amp.GradScaler(enabled=(device == 'cuda'))

    total_loss_meter = AverageMeter()
    idtri_loss_meter = AverageMeter()
    center_loss_meter = AverageMeter()
    con_loss_meter = AverageMeter()
    attn_loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    best_rank1 = 0.0

    print(
        f'Supervised contrastive loss enabled: True | '
        f'weight={args.con_loss_w:.4f} | temp={args.con_temp:.4f}'
    )

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        total_loss_meter.reset()
        idtri_loss_meter.reset()
        center_loss_meter.reset()
        con_loss_meter.reset()
        attn_loss_meter.reset()
        acc_meter.reset()

        lr_scheduler.step(epoch)
        model.train()

        for iteration, (img, pid, target_cam, labels2) in enumerate(train_loader, start=1):
            optimizer_main.zero_grad()
            optimizer_center.zero_grad()

            img = img.to(device, non_blocking=True)
            pid = pid.to(device, non_blocking=True)
            target_cam = target_cam.to(device, non_blocking=True).view(-1)
            labels2 = labels2.to(device, non_blocking=True)

            with amp.autocast(enabled=(device == 'cuda')):
                score, feat, a_vals, global_contrast_feat = model(img, pid, cam_label=target_cam)

                attn_noise = a_vals * labels2
                attn_loss = attn_noise.sum(1).mean()

                idtri_loss, center_loss, contrast_loss = loss_fun(
                    score,
                    feat,
                    pid,
                    contrast_feat=global_contrast_feat
                )

                loss = (
                    idtri_loss
                    + args.center_w * center_loss
                    + attn_loss
                    + args.con_loss_w * contrast_loss
                )

            scaler.scale(loss).backward()

            # step main optimizer
            scaler.step(optimizer_main)

            # center optimizer step
            if args.center_w > 0:
                scaler.unscale_(optimizer_center)
                for param in center_criterion.parameters():
                    if param.grad is not None:
                        param.grad.data *= (1.0 / args.center_w)
                optimizer_center.step()

            scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == pid).float().mean()
            else:
                acc = (score.max(1)[1] == pid).float().mean()

            batch_size_now = img.shape[0]
            total_loss_meter.update(loss.item(), batch_size_now)
            idtri_loss_meter.update(idtri_loss.item(), batch_size_now)
            center_loss_meter.update(center_loss.item(), batch_size_now)
            con_loss_meter.update(contrast_loss.item(), batch_size_now)
            attn_loss_meter.update(attn_loss.item(), batch_size_now)
            acc_meter.update(acc.item(), 1)

            if device == 'cuda':
                torch.cuda.synchronize()

            if iteration % 50 == 0:
                print(
                    'Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e} | '
                    'idtri={:.3f} center={:.3f} attn={:.3f} con={:.3f}'.format(
                        epoch,
                        iteration,
                        len(train_loader),
                        total_loss_meter.avg,
                        acc_meter.avg,
                        lr_scheduler._get_lr(epoch)[0],
                        idtri_loss_meter.avg,
                        center_loss_meter.avg,
                        attn_loss_meter.avg,
                        con_loss_meter.avg
                    )
                )

        print('Epoch {} finished in {:.1f}s'.format(epoch, time.time() - start_time))

        if epoch % args.eval_every == 0:
            model.eval()
            rank1, mAP = test(model, q_val_loader, g_val_loader)
            print('CMC: %.4f, mAP : %.4f' % (rank1, mAP))

            latest_path = os.path.join(
                args.output_dir,
                f'{args.Dataset_name}_camera_removed_supcon_latest.pth'
            )
            torch.save(model.state_dict(), latest_path)

            if best_rank1 < rank1:
                best_rank1 = rank1
                best_path = os.path.join(
                    args.output_dir,
                    f'{args.Dataset_name}_camera_removed_supcon_best.pth'
                )
                torch.save(model.state_dict(), best_path)
                print(f'[OK] Saved best checkpoint: {best_path}')
