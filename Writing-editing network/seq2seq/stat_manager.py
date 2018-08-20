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

    is_testing = None

    def __init__(self, config, is_testing=False):
        # TensorboardX writer object
        self.writer = SummaryWriter("saved_runs/" + config.experiment_name)

        # If this is true, then we don't write anything to the tensorboard
        StatManager.is_testing = is_testing

    class _is_test_mode:
        def __init__(self, func):
            self.func = func

        def __call__(self, *args, **kwargs):
            if not StatManager.is_testing:
                self.func(*args, **kwargs)

    @_is_test_mode
    def log_original_training_loss(self, loss, epoch):
        self.writer.add_scalar('loss/original_loss_train', loss, epoch)

    @_is_test_mode
    def log_original_validation_loss(self, loss, epoch):
        self.writer.add_scalar('loss/original_loss_valid', loss, epoch)

