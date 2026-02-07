import torch
from torch import nn
import os
import csv
import sys
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import accuracy_score, f1_score

from .utils.eval_metrics import *
from .utils.tools import *
from .model import MMIM


class Solver(object):
    """
    Disk-friendly + leakage-free protocol for grid search:

    - Train uses train split.
    - Model selection / early stopping uses ONLY dev/valid split.
    - Test is evaluated ONCE at the end using the best-by-valid weights.
    - By default, we DO NOT save per-run checkpoints during training.
      Instead, we keep the best weights in RAM (CPU) and only optionally
      save a single final checkpoint (overwriting older ones) to avoid
      filling up disk during grid search.

    How to control saving:
      - If hyp_params has attribute `save_best_ckpt` and it's False -> no file saved.
      - Otherwise -> saves ONE file at the end, overwriting any previous file:
            <repo_root>/pre_trained_models/<dataset>_best.pt

    Recommended workflow for grid search with low disk:
      1) Run grid search with `--save_best_ckpt 0` (or set hp.save_best_ckpt=False).
      2) After selecting best (lambda_diff, lambda_ctr), rerun ONE training with saving enabled
         to produce the final model file.
    """

    def __init__(self, hyp_params, train_loader, dev_loader, test_loader,
                 is_train=True, model=None, pretrained_emb=None):
        self.hp = hp = hyp_params
        self.epoch_i = 0
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.is_train = is_train
        self.model = model

        # keep for backward compatibility
        self.alpha = getattr(hp, "alpha", 0.1)
        self.beta = getattr(hp, "beta", 0.1)
        self.nce2 = getattr(hp, "nce2", 0.1)
        self.nce3 = getattr(hp, "nce3", 0.1)

        self.update_batch = hp.update_batch

        if model is None:
            self.model = model = MMIM(hp)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(self.device)

        # Criterion
        if self.hp.dataset == "ur_funny":
            self.criterion = nn.CrossEntropyLoss(reduction="mean")
        else:
            # MOSI/MOSEI regression: L1Loss == MAE
            self.criterion = nn.L1Loss(reduction="mean")

        # Optimizer (main)
        if self.is_train:
            mmilb_param, main_param, bert_param = [], [], []

            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if 'bert' in name:
                    bert_param.append(p)
                elif 'mi' in name:
                    mmilb_param.append(p)
                else:
                    main_param.append(p)

            # Xavier init for non-bert trainable params
            for p in (mmilb_param + main_param):
                if p.dim() > 1:
                    nn.init.xavier_normal_(p)

            optimizer_main_group = [
                {'params': bert_param, 'weight_decay': hp.weight_decay_bert, 'lr': hp.lr_bert},
                {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main},
            ]
            self.optimizer_main = getattr(torch.optim, self.hp.optim)(optimizer_main_group)

            # Scheduler (torch-version compatible: no verbose kw)
            self.scheduler_main = ReduceLROnPlateau(
                self.optimizer_main, mode='min', patience=hp.when, factor=0.5
            )

        # Logging
        if self.is_train:
            self.log_dir = os.path.join('logs', self.hp.dataset)
            os.makedirs(self.log_dir, exist_ok=True)

            now_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
            # include key hyperparams to disambiguate runs
            tag = f"seed{hp.seed}_aux{getattr(hp,'aux_weight','NA')}_ld{getattr(hp,'lambda_diff','NA')}_lc{getattr(hp,'lambda_ctr','NA')}"
            self.log_txt_path = os.path.join(self.log_dir, f'log_{now_time}_{tag}.txt')
            self.log_csv_path = os.path.join(self.log_dir, f'metrics_{now_time}_{tag}.csv')

            print(f"Training logs will be saved to: {self.log_dir}")

            with open(self.log_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # IMPORTANT: only train/valid metrics per epoch (no test)
                writer.writerow([
                    'Epoch',
                    'Train_Loss',
                    'Valid_Loss',
                    'Valid_MAE',
                    'Valid_Corr',
                    'Valid_Acc_2',
                    'Valid_F1',
                    'Best_Valid_Loss'
                ])

            self.print_log(f"Hyperparameters: {self.hp}")

    def print_log(self, msg: str):
        print(msg)
        if self.is_train:
            with open(self.log_txt_path, 'a') as f:
                f.write(msg + '\n')

    def _move_batch_to_device(self, batch):
        # batch format in this repo:
        # text, visual, vlens, audio, alens, y, l, bert_sent, bert_sent_type, bert_sent_mask, ids
        text, visual, vlens, audio, alens, y, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch
        dd = self.device
        text = text.to(dd)
        visual = visual.to(dd)
        audio = audio.to(dd)
        y = y.to(dd)
        l = l.to(dd)
        bert_sent = bert_sent.to(dd)
        bert_sent_type = bert_sent_type.to(dd)
        bert_sent_mask = bert_sent_mask.to(dd)
        return text, visual, vlens, audio, alens, y, l, bert_sent, bert_sent_type, bert_sent_mask, ids

    def train_and_eval(self):
        model = self.model
        optimizer_main = self.optimizer_main
        scheduler_main = self.scheduler_main
        criterion = self.criterion

        def train_one_epoch(epoch: int):
            model.train()
            epoch_loss = 0.0
            proc_loss, proc_size = 0.0, 0
            start_time = time.time()
            left_batch = self.update_batch

            num_batches = max(1, self.hp.n_train // self.hp.batch_size)

            for i_batch, batch_data in enumerate(self.train_loader):
                model.zero_grad()
                optimizer_main.zero_grad()

                text, visual, vlens, audio, alens, y, l, bert_sent, bert_sent_type, bert_sent_mask, ids = \
                    self._move_batch_to_device(batch_data)

                if self.hp.dataset == "ur_funny":
                    y = y.squeeze()

                batch_size = y.size(0)

                preds, loss_diff, loss_l1, loss_l2, loss_l3, loss_log_vars = \
                    model(text, visual, audio, vlens, alens, bert_sent, bert_sent_type, bert_sent_mask, y)

                task_loss = criterion(preds, y)

                if self.hp.contrast:
                    aux_weight = self.hp.aux_weight
                    loss = task_loss + aux_weight * (
                        self.hp.lambda_diff * loss_diff +
                        self.hp.lambda_ctr * (loss_l1 + loss_l2 + loss_l3)
                    )
                else:
                    loss = task_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    self.print_log(f"!!! FATAL: Total loss is NaN/Inf at batch {i_batch}.")
                    sys.exit(1)

                loss.backward()

                left_batch -= 1
                if left_batch == 0:
                    left_batch = self.update_batch
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)
                    optimizer_main.step()

                epoch_loss += loss.item() * batch_size
                proc_loss += loss.item() * batch_size
                proc_size += batch_size

                if i_batch % self.hp.log_interval == 0 and i_batch > 0:
                    avg_loss = proc_loss / max(1, proc_size)
                    elapsed_time = time.time() - start_time
                    self.print_log(
                        'Epoch {:2d} | Batch {:3d}/{:3d} | Time {:5.2f}ms | Loss {:5.4f} | '
                        'Task {:.3f} Diff {:.3f} L1 {:.3f} L2 {:.3f} L3 {:.3f}'.format(
                            epoch, i_batch, num_batches,
                            elapsed_time * 1000 / self.hp.log_interval,
                            avg_loss, task_loss.item(), loss_diff.item(),
                            loss_l1.item(), loss_l2.item(), loss_l3.item()
                        )
                    )
                    proc_loss, proc_size = 0.0, 0
                    start_time = time.time()

            return epoch_loss / max(1, self.hp.n_train)

        def evaluate(split: str):
            model.eval()
            loader = self.dev_loader if split == 'valid' else self.test_loader
            total_loss = 0.0
            results, truths = [], []

            with torch.no_grad():
                for batch in loader:
                    text, vision, vlens, audio, alens, y, lengths, bert_sent, bert_sent_type, bert_sent_mask, ids = batch
                    dd = self.device
                    text = text.to(dd)
                    audio = audio.to(dd)
                    vision = vision.to(dd)
                    y = y.to(dd)
                    bert_sent = bert_sent.to(dd)
                    bert_sent_type = bert_sent_type.to(dd)
                    bert_sent_mask = bert_sent_mask.to(dd)

                    if self.hp.dataset == 'iemocap':
                        y = y.long()
                    if self.hp.dataset == 'ur_funny':
                        y = y.squeeze()

                    batch_size = y.size(0)

                    preds, _, _, _, _, _ = model(
                        text, vision, audio, vlens, alens, bert_sent, bert_sent_type, bert_sent_mask
                    )

                    total_loss += criterion(preds, y).item() * batch_size
                    results.append(preds.detach().cpu())
                    truths.append(y.detach().cpu())

            denom = self.hp.n_valid if split == 'valid' else self.hp.n_test
            avg_loss = total_loss / max(1, denom)
            results = torch.cat(results)
            truths = torch.cat(truths)
            return avg_loss, results, truths

        def compute_regression_metrics(results: torch.Tensor, truths: torch.Tensor):
            r = results.numpy().reshape(-1)
            t = truths.numpy().reshape(-1)

            mae = float(np.mean(np.abs(r - t)))
            try:
                corr = float(np.corrcoef(r, t)[0][1])
            except Exception:
                corr = float("nan")

            preds_binary = (r >= 0)
            truths_binary = (t >= 0)
            acc2 = float(accuracy_score(truths_binary, preds_binary))
            f1_2 = float(f1_score(truths_binary, preds_binary, average='weighted'))

            return mae, corr, acc2, f1_2

        # --- main loop ---
        patience = self.hp.patience
        best_valid_loss = float('inf')
        best_epoch = -1

        # Keep best weights in RAM (CPU) to avoid disk usage during grid search
        best_state_dict_cpu = None

        for epoch in range(1, self.hp.num_epochs + 1):
            start = time.time()
            train_loss = train_one_epoch(epoch)
            valid_loss, valid_results, valid_truths = evaluate(split='valid')

            duration = time.time() - start
            scheduler_main.step(valid_loss)

            self.print_log("-" * 50)
            self.print_log(
                'Epoch {:2d} | Time {:5.4f} sec | Train Loss {:5.4f} | Valid Loss {:5.4f}'.format(
                    epoch, duration, train_loss, valid_loss
                )
            )
            self.print_log("-" * 50)

            # valid metrics (for proper selection/plots)
            if self.hp.dataset in ["mosi", "mosei", "mosei_senti", "sims"]:
                v_mae, v_corr, v_acc2, v_f1 = compute_regression_metrics(valid_results, valid_truths)
            else:
                v_mae, v_corr, v_acc2, v_f1 = float("nan"), float("nan"), float("nan"), float("nan")

            with open(self.log_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    f"{train_loss:.4f}",
                    f"{valid_loss:.4f}",
                    f"{v_mae:.4f}",
                    f"{v_corr:.4f}",
                    f"{v_acc2:.4f}",
                    f"{v_f1:.4f}",
                    f"{best_valid_loss:.4f}",
                ])

            # Early stopping + selection ONLY on valid
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                patience = self.hp.patience

                # save best weights to CPU RAM
                best_state_dict_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                self.print_log(
                    f"*** New Best (by VALID) Epoch: {epoch} | Valid_Loss(MAE): {best_valid_loss:.4f} ***"
                )
            else:
                patience -= 1
                if patience <= 0:
                    self.print_log("Early stopping triggered.")
                    break

        self.print_log(f'Best epoch (by VALID): {best_epoch}, best_valid_loss={best_valid_loss:.4f}')

        # Load best-by-valid weights before final TEST
        if best_state_dict_cpu is not None:
            model.load_state_dict(best_state_dict_cpu, strict=True)
            self.print_log("Loaded best-by-valid weights from RAM for final TEST eval.")
        else:
            self.print_log("[WARN] No best state captured; using current model weights for final TEST eval.")

        # Final TEST (evaluate once)
        test_loss, test_results, test_truths = evaluate(split='test')

        if self.hp.dataset in ["mosei_senti", "mosei"]:
            test_dict = eval_mosei_senti(test_results, test_truths, True)
        elif self.hp.dataset in ["mosi", "sims"]:
            test_dict = eval_mosi(test_results, test_truths, True)
        else:
            test_dict = {"test_loss": float(test_loss)}

        self.print_log("========== FINAL TEST (best-by-valid) ==========")
        self.print_log(f"Test_Loss(MAE): {test_loss:.4f}")
        self.print_log(f"{test_dict}\n")

        # Save ONE checkpoint file (optional; overwrite) to avoid disk blow-up
        save_best_ckpt = getattr(self.hp, "save_best_ckpt", True)
        save_best_ckpt = False
        if save_best_ckpt:
            # absolute repo root: <...>/src/.. -> repo root
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            ckpt_dir = os.path.join(repo_root, "pre_trained_models")
            os.makedirs(ckpt_dir, exist_ok=True)

            final_name = f"{self.hp.dataset}_best.pt"  # overwrite each run
            final_path = os.path.join(ckpt_dir, final_name)

            # also remove older per-run best files if you created them earlier
            # keep only the single final file
            try:
                for fn in os.listdir(ckpt_dir):
                    if fn.startswith(f"{self.hp.dataset}_best_") and fn.endswith(".pt"):
                        try:
                            os.remove(os.path.join(ckpt_dir, fn))
                        except Exception:
                            pass
            except Exception:
                pass

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": best_epoch,
                    "best_valid_loss": best_valid_loss,
                    "hyperparams": str(self.hp),
                },
                final_path
            )
            self.print_log(f"[SAVED] Final best model checkpoint (overwritten): {final_path}")
        else:
            self.print_log("[SKIP] save_best_ckpt=False, not saving checkpoint to disk.")

        # Write final summary file next to metrics csv
        final_path = self.log_csv_path.replace("metrics_", "final_test_").replace(".csv", ".txt")
        with open(final_path, "w") as f:
            f.write(f"Best epoch (by valid): {best_epoch}\n")
            f.write(f"Best valid loss: {best_valid_loss}\n")
            f.write(f"Test loss (MAE): {test_loss}\n")
            f.write(str(test_dict) + "\n")

        sys.stdout.flush()
