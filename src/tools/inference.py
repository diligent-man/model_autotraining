import json
from box import Box
from tqdm import tqdm
from typing import List
from src.utils.logger import Logger
from src.utils.utils import init_metrics, init_model

import torch

from torch.utils.data import DataLoader
from torch.nn.functional import sigmoid, softmax


__all__ = ["inference"]


def inference(options: Box, checkpoint_path: str, log_path: str, test_loader: DataLoader, num_threshold: int = 9) -> None:
    # Preliminary setups
    logger = Logger(phase="test")
    json_obj_lst = []
    device = "cuda" if options.MISC.CUDA else "cpu"
    checkpoint = torch.load(f=checkpoint_path, map_location=device)
    model = init_model(device=device, pretrained=False, base=options.MODEL.BASE, name=options.MODEL.NAME, state_dict=checkpoint["model_state_dict"], **options.MODEL.ARGS)

    # Start inferring with different thresholds
    with torch.no_grad():
        for threshold in torch.arange(0, 1, round(1/(num_threshold+1), 2)):
            if threshold == 0:
                # Threshold = 0 and 1 are ignored
                continue
            else:
                print("Testing with threshold =", round(threshold.item(), 2))

                # Set metrics' threshold
                # Metric not use threshold must be relegated after the ones using threshold in argument
                for i in range(len(options.METRICS.NAME_LIST)-options.METRICS.NOT_USING_THRESHOLD_METRICS):
                    options.METRICS.ARGS[str(i)].threshold = threshold

                metrics = init_metrics(name_lst=options.METRICS.NAME_LIST, args=options.METRICS.ARGS, device=device)

                # Loop over batches
                for index, batch in tqdm(enumerate(test_loader), total=len(test_loader), colour="cyan", desc="Testing"):
                    imgs, labels = batch[0].type(torch.FloatTensor).to(device), batch[1].to(device)

                    # forward pass
                    pred_labels = model(imgs)
                    if options.MODEL.ARGS.num_classes == 1:
                        # Shape: N1 -> N
                        pred_labels = sigmoid(pred_labels).squeeze(dim=1)
                    else:
                        # Shape: NC -> N
                        pred_labels = softmax(pred_labels, dim=1)

                    metrics = [metric.update(pred_labels, labels) for metric in metrics]

                # Compute metrics & Resolve output datatype returns from different metrics
                for i in range(len(options.METRICS.NAME_LIST)):
                    metrics[i] = metrics[i].compute()

                    if isinstance(metrics[i], torch.Tensor):
                        metrics[i] = metrics[i].item() if metrics[i].dim() == 1 else metrics[i].detach().cpu().numpy().tolist()

                    elif isinstance(metrics[i], tuple):
                        metrics[i] = [ele.detach().cpu().numpy().tolist() for ele in metrics[i]]

            json_obj = json.dumps(obj={
                **{"Dataset": options.DATA.DATASET_NAME,
                   "Checkpoint name": options.CHECKPOINT.NAME,
                   "Threshold": round(threshold.item(), 2)
                   },
                **{name: metric for name, metric in zip(options.METRICS.NAME_LIST, metrics)}
            }, indent=4)
            json_obj_lst.append(json_obj)

    # Logging
    log_info = ""
    for json_obj in json_obj_lst:
        log_info = log_info + json_obj + ",\n"
    log_info = "[\n" + log_info[:-2] + "\n]"  # ignore last line and comma
    logger.write(log_path, log_info)
    return None
