import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import torch.nn as nn

selected_d = {"outs": [], "trg": []}
class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, fold_id,
                 valid_data_loader=None, label_scale=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config, fold_id)
        self.config = config
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = optimizer
        self.log_step = 4 #int(data_loader.batch_size) * 1  # reduce this if you want more logs

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

        self.fold_id = fold_id
        self.selected = 0
        self.label_scale = label_scale

    def _train_epoch(self, epoch, total_epochs):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
               total_epochs: Integer, the total number of epoch
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        overall_outs = []
        overall_trgs = []
        for batch_idx, (data0,data1,data2,data3,target) in enumerate(self.data_loader):
            data0,data1,data2,data3 = data0.to(self.device), data1.to(self.device),data2.to(self.device),data3.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data0,data1,data2,data3)

            loss = self.criterion(output, target, self.device)

            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target,self.label_scale))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} '.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                ))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log, outs, trgs = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if val_log["accuracy"] > self.selected:
                self.selected = val_log["accuracy"]
                selected_d["outs"] = outs
                selected_d["trg"] = trgs
            if epoch == total_epochs:
                overall_outs.extend(selected_d["outs"])
                overall_trgs.extend(selected_d["trg"])

            if epoch == 30:
                for g in self.lr_scheduler.param_groups:
                    g['lr'] = 0.00008
            if epoch == 100
                for g in self.lr_scheduler.param_groups:
                    g['lr'] = 0.00005
            if epoch == 150:
                for g in self.lr_scheduler.param_groups:
                    g['lr'] = 0.000005

        return log, overall_outs, overall_trgs

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            for batch_idx, (data0,data1,data2,data3,target) in enumerate(self.valid_data_loader):
                # self.logger.debug('#length:{}'.format(
                #         len(self.valid_data_loader)
                #     ))
                data0,data1,data2,data3 = data0.to(self.device), data1.to(self.device),data2.to(self.device),data3.to(self.device)
                target = target.to(self.device)
                output = self.model(data0,data1,data2,data3)
                loss = self.criterion(output, target, self.device)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target,self.label_scale))
                 
                if batch_idx % self.log_step == 0:   
                    self.logger.debug('#Test Epoch: {} {} Loss: {:.6f} '.format(
                        epoch,
                        self._progress_val(batch_idx),
                        loss.item(),
                    ))
                
                preds_ = output.data.max(1, keepdim=True)[1].cpu()

                outs = np.append(outs, preds_.cpu().numpy())
                trgs = np.append(trgs, target.data.cpu().numpy())


        return self.valid_metrics.result(), outs, trgs

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    
    def _progress_val(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.valid_data_loader, 'n_samples'):
            current = batch_idx * self.valid_data_loader.batch_size
            total = self.valid_data_loader.n_samples
        else:
            current = batch_idx
            total = len(self.valid_data_loader)
        return base.format(current, total, 100.0 * current / total)
