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
    predict_right_after = 3
    patience = 5

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

class LargeDataset(CommonConfig):
    relative_data_path = '/data/large-json/train.dat'
    relative_dev_path = '/data/large-json/dev.dat'
    relative_test_path = '/data/large-json/test.dat'
    relative_gen_path = '/data/large-json/fake%d.dat'

class SmallTopicsConfig1(SmallDatasetWithTopics):
    emsize = 512
    context_dim = 128
    lr = 0.0001
    pretrained = None
    use_topics = False
    experiment_name = "smt-lr-0.0001"

class SmallTopicsConfig2(SmallDatasetWithTopics):
    emsize = 512
    context_dim = 128
    lr = 0.0001
    pretrained = 'embeddings/complete-512.vec'
    use_topics = False
    experiment_name = "smt-lr-0.0001-WE-512"

class SmallTopicsConfig3(SmallDatasetWithTopics):
    emsize = 300
    context_dim = 128
    lr = 0.0001
    pretrained = 'embeddings/complete.vec'
    use_topics = False
    experiment_name = "smt-lr-0.0001-WE-300"

class SmallTopicsConfig4(SmallDatasetWithTopics):
    emsize = 512
    context_dim = 128
    lr = 0.0001
    pretrained = None
    use_topics = True
    experiment_name = "smt-with-topics-lr-0.0001"

class SmallTopicsConfig5(SmallDatasetWithTopics):
    emsize = 512
    context_dim = 128
    lr = 0.0001
    pretrained = 'embeddings/complete-512.vec'
    use_topics = True
    experiment_name = "smt-with-topics-lr-0.0001-WE-512"

class SmallTopicsConfig6(SmallDatasetWithTopics):
    emsize = 300
    context_dim = 128
    lr = 0.0001
    pretrained = 'embeddings/complete.vec'
    use_topics = True
    experiment_name = "smt-with-topics-lr-0.0001-WE-300"


class SmallConfig1(SmallDataset):
    emsize = 512
    context_dim = 128
    lr = 0.0001
    pretrained = None
    use_topics = False
    experiment_name = "sm-lr-0.0001"

class SmallConfig2(SmallDataset):
    emsize = 512
    context_dim = 128
    lr = 0.0001
    pretrained = 'embeddings/complete-512.vec'
    use_topics = False
    experiment_name = "sm-lr-0.0001-WE-512"

class SmallConfig3(SmallDataset):
    emsize = 300
    context_dim = 128
    lr = 0.0001
    pretrained = 'embeddings/complete.vec'
    use_topics = False
    experiment_name = "sm-lr-0.0001-WE-300"


class LargeConfig1(LargeDataset):
    emsize = 512
    context_dim = 128
    lr = 0.0001
    pretrained = None
    use_topics = False
    experiment_name = "lg-lr-0.0001"

class LargeConfig2(LargeDataset):
    emsize = 512
    context_dim = 128
    lr = 0.0001
    pretrained = 'embeddings/complete-512.vec'
    use_topics = False
    experiment_name = "lg-lr-0.0001-WE-512"

class LargeConfig3(LargeDataset):
    emsize = 300
    context_dim = 128
    lr = 0.0001
    pretrained = 'embeddings/complete.vec'
    use_topics = False
    experiment_name = "lg-lr-0.0001-WE-300"

class LargeConfig4(LargeDataset):
    emsize = 512
    context_dim = 128
    lr = 0.0001
    pretrained = None
    use_topics = True
    experiment_name = "lg-with-topics-lr-0.0001"

class LargeConfig5(LargeDataset):
    emsize = 512
    context_dim = 128
    lr = 0.0001
    pretrained = 'embeddings/complete-512.vec'
    use_topics = True
    experiment_name = "lg-with-topics-lr-0.0001-WE-512"

class LargeConfig6(LargeDataset):
    emsize = 300
    context_dim = 128
    lr = 0.0001
    pretrained = 'embeddings/complete.vec'
    use_topics = True
    experiment_name = "lg-with-topics-lr-0.0001-WE-300"

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
                 "l6": LargeConfig6()}

def get_conf(name):
    return configuration[name]