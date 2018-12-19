class CommonConfig(object):
    cell = "GRU"
    nlayers = 1
    batch_size = 20
    validation_batch_size = 32
    data_parallel = False
    distributed_data_parallel = False
    dropout = 0
    epochs = 20
    bidirectional = True
    max_grad_norm = 10
    min_freq = 5
    num_exams = 3
    log_interval = 50
    patience = 5
    print_running_loss = False

    #Common Configuration that is usually played with.
    # Will remain the same unless overridden
    emsize = 512
    context_dim = 128
    lr = 0.0001
    pretrained = None
    use_topics = False
    use_labels = False
    use_intra_attention = False
    window_size_attention = 3

class Dataset(CommonConfig):
    relative_data_path = '/data/structurally-labelled-data/tf-idf/train.dat'
    relative_dev_path = '/data/structurally-labelled-data/tf-idf/dev.dat'
    relative_test_path = '/data/structurally-labelled-data/tf-idf/test.dat'

class BaselineConfig(Dataset):
    experiment_name = "lg-lr-0.0001"
    pretrained = 'embeddings/complete-512.vec'

class TopicConfig(Dataset):
    use_topics = True
    pretrained = 'embeddings/complete-512.vec'
    experiment_name = "lg-with-topics-lr-0.0001"

class LabelConfig(Dataset):
    use_labels = True
    pretrained = 'embeddings/complete-512.vec'
    experiment_name = "lg-with-labels-lr-0.0001"

class Label_Topics_Config(Dataset):
    use_topics = True
    use_labels = True
    pretrained = 'embeddings/complete-512.vec'
    experiment_name = "lg-with-labels-and-topics-lr-0.0001"

class Labels_Topics_IntraAttention_Config(Dataset):
    use_topics = True
    use_labels = True
    use_intra_attention = True
    window_size_attention = 5
    experiment_name = "lg-with-labels-and-tf-idf-lr-0.0001"
    pretrained = 'embeddings/complete-512.vec'

def print_config(config):
    print("[batch_size = {}, dataparallel = {}, epochs = {}, log_interval = {}, patience = {}, relative_data_path = {}, relative_dev_path = {}, emsize = {},context_dim = {}, learning rate = {}, pretrained = {}, use_topics = {}, use_structure_labels = {}, experiment_name = {}]".format(config.batch_size
          , config.dataparallel, config.epochs, config.log_interval, config.patience,
          config.relative_data_path, config.relative_dev_path, config.emsize, config.context_dim,
          config.lr, config.pretrained, config.use_topics, config.use_labels, config.experiment_name), flush=True)

configuration = {
                 "b": BaselineConfig(),
                 "t": TopicConfig(),
                 "l": LabelConfig(),
                 "lt": Label_Topics_Config(),
                 "lti": Labels_Topics_IntraAttention_Config()}

def get_conf(name):
    print("Config name is {}".format(name))
    return configuration[name]