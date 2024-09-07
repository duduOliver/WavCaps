#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# only record main process logs on wandb
# Don't forget to put log_results funtion under the condition check of main process

import time
from pprint import PrettyPrinter
import wandb
import torch
import argparse
import ruamel.yaml as yaml
from tqdm import tqdm
from loguru import logger
from data_handling.datamodule import AudioCaptionDataModule
from data_handling.pretrain_dataset import pretrain_dataloader
from models.ase_model import ASE
import torch.distributed as dist
from tools.optim_utils import get_optimizer, cosine_lr
from tools.utils import (
    get_rank,
    get_world_size,
    # init_distributed_mode,
    # is_dist_avail_and_initialized,
    # is_main_process,
    setup_seed,
    AverageMeter, t2a, a2t, set_logger, log_results,
)

WB_LOG = True

def train(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
     
    if model.audio_encoder.config["audio_encoder_args"]["type"] == "dac":
            # Set the codec part to evaluation mode
            model.audio_encoder.eval()
    # print(f"model.audio_encoder in eval mode: {not model.audio_encoder.training}")
    elif model.audio_encoder.config["audio_encoder_args"]["type"] == "dac_embedder":
            # Set the codec part to evaluation mode
            model.audio_encoder.codes_enc.eval()
    elif model.audio_encoder.config["audio_encoder_args"]["type"] == "dac_htsat":
            # Set the codec part to evaluation mode
            # model.audio_encoder.codes_enc.to(device)
            model.audio_encoder.codes_enc.eval()
    elif model.audio_encoder.config["audio_encoder_args"]["type"] == "vamp":
            # Set the codec part to evaluation mode
            # model.audio_encoder.codes_enc.to(device)
            model.audio_encoder.codes_enc.eval()

    epoch_loss = AverageMeter()
    start_time = time.time()

    for batch_id, (audio, text, idx) in tqdm(enumerate(dataloader), total=len(dataloader)):

        optimizer.zero_grad()

        step = len(dataloader) * (epoch - 1) + batch_id
        scheduler(step)
        if WB_LOG:
            wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=step)

        audio = audio.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)

        loss = model(audio, text, idx)

        loss.backward()
        optimizer.step()

        epoch_loss.update(loss.cpu().item())

    elapsed_time = time.time() - start_time

    if WB_LOG:
        wandb.log({"loss": epoch_loss.avg,
               "epoch": epoch})

    return {
        "loss": epoch_loss.avg,
        "time": elapsed_time
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="settings/baseline.yaml", type=str,
                        help="Setting files")
    parser.add_argument("-n", "--exp_name", default="exp_name", type=str,
                        help="name of this experiment.")
    parser.add_argument("-l", "--lr", default=5e-5, type=float,
                        help="Learning rate.")
    parser.add_argument("-t", "--model_type", default="cnn", type=str,
                        help="Model type.")
    parser.add_argument("-m", "--model", default="Cnn14", type=str,
                        help="Model name.")
    parser.add_argument("-a", "--max_length", default=30, type=int,
                        help="Max length.")
    parser.add_argument("-s", "--batch_size", default=128, type=int,
                        help="Batch size.")
    parser.add_argument("-b", "--blacklist", default='blacklist_exclude_ub8k_esc50_vggsound.json', type=str,
                        help="Blacklist file.")
    args = parser.parse_args()

    exp_name = args.exp_name

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"])

    # setup seed
    seed = config["seed"] + get_rank()
    setup_seed(seed)

    exp_name = exp_name + "_AE_" + config["audio_encoder_args"]["model"] + "_" + str(config["audio_encoder_args"]["freeze"]) + "_TE_" + config["text_encoder_args"]["type"] + "_" + str(config["text_encoder_args"]["freeze"]) + f"_lr_{args.lr}_seed_{seed}"

    if WB_LOG:
        wandb.init(
            project="DAC_TEST",
            name=exp_name,
            config=config,
            group="TESTING",  # all runs for the experiment in one group
        )

    # self_add create BASELINE dataloader
    clotho_datamodule = AudioCaptionDataModule(config, "Clotho")
    dataloader = clotho_datamodule.train_dataloader(num_tasks=get_world_size(),
                                                    global_rank=get_rank())
    # setup model
    model = ASE(config)
    model = model.to(device)
    # setup optim utils
    optimizer = get_optimizer(model.parameters(),
                              lr=config["optim_args"]["lr"],
                              betas=config["optim_args"]["betas"],
                              eps=config["optim_args"]["eps"],
                              momentum=config["optim_args"]["momentum"],
                              optimizer_name=config["optim_args"]["optimizer_name"])
    scheduler = cosine_lr(optimizer,
                          base_lr=config["optim_args"]["lr"],
                          warmup_length=config["optim_args"]["warmup_epochs"] * len(dataloader),
                          steps=len(dataloader) * config["training"]["epochs"])
    start_epoch = 1
    max_epoch = config["training"]["epochs"]

    if config["resume"]:
        cp = torch.load(config.checkpoint, map_location="cpu")
        state_dict = cp["model"]

        optimizer.load_state_dict(cp["optimizer"])
        start_epoch = cp["epoch"] + 1
        model.load_state_dict(state_dict)

    # setup logger
    model_output_dir, log_output_dir = set_logger(exp_name)

    main_logger = logger.bind(indent=1)

    # print training settings
    printer = PrettyPrinter()
    main_logger.info('Training setting:\n'
                     f'{printer.pformat(config)}')

    main_logger.info(f'Total numer of parameters: {sum([i.numel() for i in model.parameters()])}')
    main_logger.info(f'Size of training set: {len(dataloader.dataset)}, size of batches: {len(dataloader)}')

    if WB_LOG:
        wandb.watch(model)

    clotho_val_loader = clotho_datamodule.val_dataloader()

    loss_stats = []
    ac_recall_stats = []
    clotho_recall_stats = []
    for epoch in range(start_epoch, max_epoch + 1):
        main_logger.info(f'Training for epoch [{epoch}]')

        train_statics = train(model, dataloader, optimizer, scheduler, device, epoch)
        loss = train_statics["loss"]
        elapsed_time = train_statics["time"]
        loss_stats.append(loss)

        main_logger.info(f'Training statistics:\tloss for epoch [{epoch}]: {loss:.3f},'
                         f'\ttime: {elapsed_time:.1f}, lr: {optimizer.param_groups[0]["lr"]:.6f}.')

        clotho_metrics = validate(model, clotho_val_loader, device) # model_without_ddp
        if WB_LOG:
            log_results(clotho_metrics, 'Clotho', main_logger, test=False)
        clotho_recall_stats.append(clotho_metrics["t2a"][0] + clotho_metrics["a2t"][0])
        if clotho_recall_stats[-1] >= max(clotho_recall_stats):
            sav_obj = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "epoch": epoch
            }
            torch.save(sav_obj, str(model_output_dir) + "/clotho_best_model.pt")

    main_logger.info('Evaluation start...')
    clotho_test_loader = clotho_datamodule.test_dataloader()
    model.load_state_dict(torch.load(str(model_output_dir) + "/clotho_best_model.pt")["model"])
    main_logger.info(f"Evaluation best Clotho model... epoch:{torch.load(str(model_output_dir) + '/clotho_best_model.pt')['epoch']}")
    clotho_metrics = validate(model, clotho_test_loader, device) # model_without_ddp
    if WB_LOG:
        log_results(clotho_metrics, 'Clotho', main_logger, test=True)
        main_logger.info("Done.")
        wandb.finish()


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    audio_embeds_all, text_embeds_all = [], []
    for batch_idx, (audio, text, idx) in tqdm(enumerate(dataloader), total=len(dataloader)):
        audio = audio.to(device)

        audio_embeds = model.encode_audio(audio)
        text_embeds = model.encode_text(text)

        audio_embeds_all.append(audio_embeds.cpu())
        text_embeds_all.append(text_embeds.cpu())

    audio_embeds_all = torch.cat(audio_embeds_all, dim=0).numpy()
    text_embeds_all = torch.cat(text_embeds_all, dim=0).numpy()

    # evaluate text to audio retrieval
    r1, r5, r10, r50, medr, meanr, mAP10 = t2a(audio_embeds_all, text_embeds_all)

    # evaluate audio to text retrieval
    r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP10_a = a2t(audio_embeds_all, text_embeds_all)

    return {"t2a": [r1, r5, r10, r50, medr, meanr, mAP10],
            "a2t": [r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP10_a]}


if __name__ == '__main__':
    main()