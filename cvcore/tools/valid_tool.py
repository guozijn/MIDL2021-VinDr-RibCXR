import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from cvcore.utils import save_checkpoint
from sklearn.metrics import f1_score, recall_score, precision_score
def valid_model(_print, cfg, model, valid_loader,
                loss_function, metric_function, epoch,
                best_metric=None, checkpoint=False):
    # switch to evaluate mode
    model.eval()
    if hasattr(metric_function, "reset"):
        metric_function.reset()
    preds = []
    labels = []
    tbar = tqdm(valid_loader)

    with torch.no_grad():
        for i, (image, label) in enumerate(tbar):
            image = image.to(device='cuda', dtype=torch.float, non_blocking=True)
            output = model(image)
            w_output = torch.sigmoid(output) > 0.5
            preds.append((output.cpu(), w_output.cpu()))
            labels.append(label.cpu())
    metric_name= cfg.METRIC.NAME
    logits, preds = zip(*preds)
    logits, preds, labels = torch.cat(logits, 0), torch.cat(preds, 0), torch.cat(labels, 0)
    val_loss = loss_function(logits.float(), labels)
    final_score_tensor = metric_function(preds, labels.long())
    if hasattr(metric_function, "aggregate"):
        final_score_tensor = metric_function.aggregate()
    final_score = torch.nanmean(final_score_tensor.float()).item()
    if hasattr(metric_function, "reset"):
        metric_function.reset()

    best_metric_log = best_metric if best_metric is not None else final_score
    _print(f"Validation {metric_name}: {final_score:04f}, val loss:{val_loss:05f} best: {best_metric_log:04f}\n")


    # checkpoint

    if checkpoint:
        is_best = final_score > best_metric
        best_metric = max(final_score, best_metric)
        save_dict = {"epoch": epoch + 1,
                     "arch": cfg.NAME,
                     "state_dict": model.state_dict(),
                     "best_metric": best_metric}
        save_filename = f"{cfg.NAME}.pth"
        if is_best: # only save best checkpoint, no need resume
            print("score improved, saving new checkpoint...")
            save_checkpoint(save_dict, is_best,
                            root=cfg.DIRS.WEIGHTS, filename=save_filename)
    return val_loss.item(), final_score, best_metric
