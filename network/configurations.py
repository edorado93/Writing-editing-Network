class CommonConfig(object):
    cell = "GRU"
    nlayers = 1
    batch_size = 20
    validation_batch_size = 32
    data_parallel = True
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
    use_intra_attention = True
    window_size_attention = 3

class ThreeMostRelevantDataset(CommonConfig):
    relative_data_path = '/data/large-json-three-most-relevant/train.dat'
    relative_dev_path = '/data/large-json-three-most-relevant/dev.dat'
    relative_test_path = '/data/large-json-three-most-relevant/test.dat'

class TfIdfAdditionalTopicDataset(CommonConfig):
    relative_data_path = '/data/large-json-additional-topics-tf-idf/train.dat'
    relative_dev_path = '/data/large-json-additional-topics-tf-idf/dev.dat'
    relative_test_path = '/data/large-json-additional-topics-tf-idf/test.dat'

class LargeDataset(CommonConfig):
    relative_data_path = '/data/large-json/train.dat'
    relative_dev_path = '/data/large-json/dev.dat'
    relative_test_path = '/data/large-json/test.dat'

class PubMedStateDiagramDataset(CommonConfig):
    relative_data_path = '/data/structurally-labelled-data/pub-med-state-diagram/train.dat'
    relative_dev_path = '/data/structurally-labelled-data/pub-med-state-diagram/dev.dat'
    relative_test_path = '/data/structurally-labelled-data/pub-med-state-diagram/test.dat'

class PubMedAndTfIdfDataset(CommonConfig):
    relative_data_path = '/data/structurally-labelled-data/tf-idf/train.dat'
    relative_dev_path = '/data/structurally-labelled-data/tf-idf/dev.dat'
    relative_test_path = '/data/structurally-labelled-data/tf-idf/test.dat'

class LargeConfig1(LargeDataset):
    experiment_name = "lg-lr-0.0001"

class LargeConfig2(LargeDataset):
    pretrained = 'embeddings/complete-512.vec'
    experiment_name = "lg-lr-0.0001-WE-512"

class LargeConfig3(LargeDataset):
    emsize = 300
    pretrained = 'embeddings/complete.vec'
    experiment_name = "lg-lr-0.0001-WE-300"

class LargeConfig4(LargeDataset):
    use_topics = True
    experiment_name = "lg-with-topics-lr-0.0001"

class LargeConfig5(LargeDataset):
    pretrained = 'embeddings/complete-512.vec'
    use_topics = True
    experiment_name = "lg-with-topics-lr-0.0001-WE-512"

class LargeConfig6(LargeDataset):
    emsize = 300
    pretrained = 'embeddings/complete.vec'
    use_topics = True
    experiment_name = "lg-with-topics-lr-0.0001-WE-300"

class ThreeMostRelevant1(ThreeMostRelevantDataset):
    use_topics = True
    experiment_name = "lg-with-topics-lr-0.0001-three-most-relevant"

class ThreeMostRelevant2(ThreeMostRelevantDataset):
    pretrained = 'embeddings/complete-512.vec'
    use_topics = True
    experiment_name = "lg-with-topics-lr-0.0001-WE-512-three-most-relevant"

class ThreeMostRelevant3(ThreeMostRelevantDataset):
    emsize = 300
    pretrained = 'embeddings/complete.vec'
    use_topics = True
    experiment_name = "lg-with-topics-lr-0.0001-WE-300-three-most-relevant"

class TfIdfAdditional1(TfIdfAdditionalTopicDataset):
    use_topics = True
    experiment_name = "lg-with-topics-lr-0.0001-tf-idf"

class TfIdfAdditional2(TfIdfAdditionalTopicDataset):
    pretrained = 'embeddings/complete-512.vec'
    use_topics = True
    experiment_name = "lg-with-topics-lr-0.0001-WE-512-tf-idf"

class TfIdfAdditional3(TfIdfAdditionalTopicDataset):
    emsize = 300
    pretrained = 'embeddings/complete.vec'
    use_topics = True
    experiment_name = "lg-with-topics-lr-0.0001-WE-300-tf-idf"

class LabelConfig1(PubMedStateDiagramDataset):
    use_labels = True
    experiment_name = "lg-with-labels-lr-0.0001"

class LabelConfig2(PubMedStateDiagramDataset):
    pretrained = 'embeddings/complete-512.vec'
    use_labels = True
    experiment_name = "lg-with-labels-lr-0.0001-WE-512"

class LabelConfig3(PubMedStateDiagramDataset):
    emsize = 300
    pretrained = 'embeddings/complete.vec'
    use_labels = True
    experiment_name = "lg-with-labels-lr-0.0001-WE-300"

class LabelAndTopics1(PubMedStateDiagramDataset):
    use_topics = True
    use_labels = True
    experiment_name = "lg-with-labels-and-topics-lr-0.0001"

class LabelAndTopics2(PubMedStateDiagramDataset):
    use_topics = True
    pretrained = 'embeddings/complete-512.vec'
    use_labels = True
    experiment_name = "lg-with-labels-and-topics-lr-0.0001-WE-512"

class LabelAndTopics3(PubMedStateDiagramDataset):
    use_topics = True
    emsize = 300
    pretrained = 'embeddings/complete.vec'
    use_labels = True
    experiment_name = "lg-with-labels-and-topics-lr-0.0001-WE-300"

class LabelsAndTf1(PubMedAndTfIdfDataset):
    use_topics = True
    use_labels = True
    experiment_name = "lg-with-labels-and-tf-idf-lr-0.0001"

class LabelsAndTf2(PubMedAndTfIdfDataset):
    use_topics = True
    pretrained = 'embeddings/complete-512.vec'
    use_labels = True
    experiment_name = "lg-with-labels-and-tf-idf-lr-0.0001-WE-512"

class LabelsAndTf3(PubMedAndTfIdfDataset):
    use_topics = True
    emsize = 300
    pretrained = 'embeddings/complete.vec'
    use_labels = True
    experiment_name = "lg-with-labels-and-tf-idf-lr-0.0001-WE-300"

def print_config(config):
    print("[batch_size = {}, dataparallel = {}, epochs = {}, log_interval = {}, patience = {}, relative_data_path = {}, relative_dev_path = {}, emsize = {},context_dim = {}, learning rate = {}, pretrained = {}, use_topics = {}, use_structure_labels = {}, experiment_name = {}]".format(config.batch_size
          , config.dataparallel, config.epochs, config.log_interval, config.patience,
          config.relative_data_path, config.relative_dev_path, config.emsize, config.context_dim,
          config.lr, config.pretrained, config.use_topics, config.use_labels, config.experiment_name), flush=True)

configuration = {
                 "l1": LargeConfig1(),
                 "l2": LargeConfig2(),
                 "l3": LargeConfig3(),
                 "l4": LargeConfig4(),
                 "l5": LargeConfig5(),
                 "l6": LargeConfig6(),
                 "la1": LabelConfig1(),
                 "la2": LabelConfig2(),
                 "la3": LabelConfig3(),
                 "l_and_t1": LabelAndTopics1(),
                 "l_and_t2": LabelAndTopics2(),
                 "l_and_t3": LabelAndTopics3(),
                 "three_1": ThreeMostRelevant1(),
                 "three_2": ThreeMostRelevant2(),
                 "three_3": ThreeMostRelevant3(),
                 "tf_1": TfIdfAdditional1(),
                 "tf_2": TfIdfAdditional2(),
                 "tf_3": TfIdfAdditional3(),
                 "l_and_tf1": LabelsAndTf1(),
                 "l_and_tf2": LabelsAndTf2(),
                 "l_and_tf3": LabelsAndTf3()}

def get_conf(name):
    print("Config name is {}".format(name))
    return configuration[name]