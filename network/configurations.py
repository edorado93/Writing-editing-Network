class CommonConfig(object):
    cell = "GRU"
    nlayers = 1
    batch_size = 20
    validation_batch_size = 32
    data_parallel = False
    distributed_data_parallel = False
    input_dropout_p = 0
    dropout_p = 0
    output_dropout_p = 0
    epochs = 20
    bidirectional = True
    max_grad_norm = 10
    min_freq = 5
    num_exams = 3
    log_interval = 50
    patience = 5
    print_running_loss = False
    emsize = 512
    context_dim = 128
    lr = 0.0001
    pretrained = None
    use_topics = False
    use_labels = False
    use_intra_attention = False
    window_size_attention = 3

class ACLDataset(CommonConfig):
    relative_data_path = '/data/acl/train.dat'
    relative_dev_path = '/data/acl/dev.dat'
    relative_test_path = '/data/acl/test.dat'

class XMLADataset(CommonConfig):
    relative_data_path = '/data/structurally-labelled-data/tf-idf/train.dat'
    relative_dev_path = '/data/structurally-labelled-data/tf-idf/dev.dat'
    relative_test_path = '/data/structurally-labelled-data/tf-idf/test.dat'

def get_baseline_config(data):
    class BaselineConfig(data):
        experiment_name = "lg-lr-0.0001"
        pretrained = 'embeddings/complete-512.vec'
    return BaselineConfig

def get_topic_config(data):
    class TopicConfig(data):
        use_topics = True
        pretrained = 'embeddings/complete-512.vec'
        experiment_name = "lg-with-topics-lr-0.0001"
    return TopicConfig

def get_structure_config(data):
    class StructureConfig(data):
        use_labels = True
        pretrained = 'embeddings/complete-512.vec'
        experiment_name = "lg-with-labels-lr-0.0001"
    return StructureConfig

def get_structure_and_topic_config(data):
    class Structure_Topics_Config(data):
        use_topics = True
        use_labels = True
        pretrained = 'embeddings/complete-512.vec'
        experiment_name = "lg-with-labels-and-topics-lr-0.0001"
    return Structure_Topics_Config

def get_structure_topic_and_attention_config(data):
    class Structure_Topics_IntraAttention_Config(data):
        use_topics = True
        use_labels = True
        use_intra_attention = True
        window_size_attention = 5
        experiment_name = "lg-with-labels-and-tf-idf-lr-0.0001"
        pretrained = 'embeddings/complete-512.vec'
    return Structure_Topics_IntraAttention_Config

def init(dataset):
    dataset = XMLADataset if dataset == "xmla" else ACLDataset
    return {"b": get_baseline_config(dataset),
            "t": get_topic_config(dataset),
            "s": get_structure_config(dataset),
            "st": get_structure_and_topic_config(dataset),
            "sti": get_structure_topic_and_attention_config(dataset)}
