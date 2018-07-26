class CommonConfig(object):
    cell = "GRU"
    nlayers = 1
    batch_size = 20
    dataparallel = False
    dropout = 0
    epochs = 20
    bidirectional = True
    max_grad_norm = 10
    min_freq = 5
    num_exams = 3
    log_interval = 1000
    patience = 5

    #Common Configuration that is usually played with.
    # Will remain the same unless overridden
    emsize = 512
    context_dim = 128
    lr = 0.0001
    pretrained = None
    use_topics = False
    use_labels = False

class SmallDatasetWithTopics(CommonConfig):
    relative_data_path = '/data/small-json-topics/train.dat'
    relative_dev_path = '/data/small-json-topics/dev.dat'
    relative_test_path = '/data/small-json-topics/test.dat'
    relative_gen_path = '/data/small-json-topics/fake%d.dat'

class SmallDataset(CommonConfig):
    relative_data_path = '/data/small-json/train.dat'
    relative_dev_path = '/data/small-json/dev.dat'
    relative_test_path = '/data/small-json/test.dat'
    relative_gen_path = '/data/small-json/fake%d.dat'

class ThreeMostRelevantDataset(CommonConfig):
    relative_data_path = '/data/large-json-three-most-relevant/train.dat'
    relative_dev_path = '/data/large-json-three-most-relevant/dev.dat'
    relative_test_path = '/data/large-json-three-most-relevant/test.dat'
    relative_gen_path = '/data/large-json-three-most-relevant/fake%d.dat'

class TfIdfAdditionalTopicDataset(CommonConfig):
    relative_data_path = '/data/large-json-additional-topics-tf-idf/train.dat'
    relative_dev_path = '/data/large-json-additional-topics-tf-idf/dev.dat'
    relative_test_path = '/data/large-json-additional-topics-tf-idf/test.dat'
    relative_gen_path = '/data/large-json-additional-topics-tf-idf/fake%d.dat'

class LargeDataset(CommonConfig):
    relative_data_path = '/data/temp/train.dat'
    relative_dev_path = '/data/temp/dev.dat'
    relative_test_path = '/data/temp/test.dat'
    relative_gen_path = '/data/temp/fake%d.dat'

class SmallTopicsConfig1(SmallDatasetWithTopics):
    experiment_name = "smt-lr-0.0001"

class SmallTopicsConfig2(SmallDatasetWithTopics):
    pretrained = 'embeddings/complete-512.vec'
    experiment_name = "smt-lr-0.0001-WE-512"

class SmallTopicsConfig3(SmallDatasetWithTopics):
    emsize = 300
    pretrained = 'embeddings/complete.vec'
    experiment_name = "smt-lr-0.0001-WE-300"

class SmallTopicsConfig4(SmallDatasetWithTopics):
    use_topics = True
    experiment_name = "smt-with-topics-lr-0.0001"

class SmallTopicsConfig5(SmallDatasetWithTopics):
    pretrained = 'embeddings/complete-512.vec'
    use_topics = True
    experiment_name = "smt-with-topics-lr-0.0001-WE-512"

class SmallTopicsConfig6(SmallDatasetWithTopics):
    emsize = 300
    pretrained = 'embeddings/complete.vec'
    use_topics = True
    experiment_name = "smt-with-topics-lr-0.0001-WE-300"


class SmallConfig1(SmallDataset):
    experiment_name = "sm-lr-0.0001"

class SmallConfig2(SmallDataset):
    pretrained = 'embeddings/complete-512.vec'
    experiment_name = "sm-lr-0.0001-WE-512"

class SmallConfig3(SmallDataset):
    emsize = 300
    pretrained = 'embeddings/complete.vec'
    experiment_name = "sm-lr-0.0001-WE-300"


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

class LabelConfig1(LargeDataset):
    use_labels = True
    experiment_name = "lg-with-labels-lr-0.0001"

class LabelConfig2(LargeDataset):
    pretrained = 'embeddings/complete-512.vec'
    use_labels = True
    experiment_name = "lg-with-labels-lr-0.0001-WE-512"

class LabelConfig3(LargeDataset):
    emsize = 300
    pretrained = 'embeddings/complete.vec'
    use_labels = True
    experiment_name = "lg-with-labels-lr-0.0001-WE-300"

class LabelAndTopics1(LabelConfig1):
    use_topics = True
    experiment_name = "lg-with-labels-and-topics-lr-0.0001-WE-300"

class LabelAndTopics2(LabelConfig2):
    use_topics = True
    experiment_name = "lg-with-labels-and-topics-lr-0.0001-WE-300"

class LabelAndTopics3(LabelConfig3):
    use_topics = True
    experiment_name = "lg-with-labels-and-topics-lr-0.0001-WE-300"


def print_config(config):
    print("[batch_size = {}, dataparallel = {}, epochs = {}, log_interval = {}, patience = {}, relative_data_path = {}, relative_dev_path = {}, emsize = {},context_dim = {}, learning rate = {}, pretrained = {}, use_topics = {}, use_structure_labels = {}, experiment_name = {}]".format(config.batch_size
          , config.dataparallel, config.epochs, config.log_interval, config.patience,
          config.relative_data_path, config.relative_dev_path, config.emsize, config.context_dim,
          config.lr, config.pretrained, config.use_topics, config.use_labels, config.experiment_name), flush=True)

configuration = {
                 "st1": SmallTopicsConfig1(),
                 "st2": SmallTopicsConfig2(),
                 "st3": SmallTopicsConfig3(),
                 "st4": SmallTopicsConfig4(),
                 "st5": SmallTopicsConfig5(),
                 "st6": SmallTopicsConfig6(),
                 "s1": SmallConfig1(),
                 "s2": SmallConfig2(),
                 "s3": SmallConfig3(),
                 "l1": LargeConfig1(),
                 "l2": LargeConfig2(),
                 "l3": LargeConfig3(),
                 "l4": LargeConfig4(),
                 "l5": LargeConfig5(),
                 "l6": LargeConfig6(),
                 "la1": LabelConfig1(),
                 "la2": LabelConfig1(),
                 "la3": LabelConfig1(),
                 "l_and_t1": LabelAndTopics1(),
                 "l_and_t2": LabelAndTopics2(),
                 "l_and_t3": LabelAndTopics3(),
                 "three_1": ThreeMostRelevant1(),
                 "three_2": ThreeMostRelevant2(),
                 "three_3": ThreeMostRelevant3(),
                 "tf_1": TfIdfAdditional1(),
                 "tf_2": TfIdfAdditional2(),
                 "tf_3": TfIdfAdditional3()}

def get_conf(name):
    return configuration[name]