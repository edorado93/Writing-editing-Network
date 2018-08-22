from tensorboardX import SummaryWriter

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class StatManager:
    def __init__(self, config, is_testing=False):
        # TensorboardX writer object
        self.writer = SummaryWriter("saved_runs/" + config.experiment_name)

        # If this is true, then we don't write anything to the tensorboard
        self.is_testing = is_testing

    def log_original_training_loss(self, loss, epoch):
        if not self.is_testing:
            self.writer.add_scalar('loss/original_loss_train', loss, epoch)

    def log_original_validation_loss(self, loss, epoch):
        if not self.is_testing:
            self.writer.add_scalar('loss/original_loss_valid', loss, epoch)

