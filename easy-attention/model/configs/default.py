import ml_collections


def get_config():
    # return normal_config()
    return small_config()


def normal_config():
    config = ml_collections.ConfigDict()
    config.dataset_batch_size = 16
    config.dataset_sequence_length = 256
    config.learning_rate = 0.01
    config.momentum = 0.9
    config.model_embed_dim = 128
    config.transformer_num_layers = 6
    config.transformer_num_attention_heads = 8
    config.transformer_attention_size_per_head = 4
    config.transformer_dropout_rate = 0.1
    return ml_collections.FrozenConfigDict(config)


def small_config():
    config = ml_collections.ConfigDict()
    config.dataset_batch_size = 4096
    config.dataset_sequence_length = 256
    config.learning_rate = 1.0
    config.momentum = 0.9
    config.model_embed_dim = 32
    config.transformer_num_layers = 4
    config.transformer_num_attention_heads = 4
    config.transformer_attention_size_per_head = 4
    config.transformer_dropout_rate = 0.1
    return ml_collections.FrozenConfigDict(config)


def tiny_config():
    config = ml_collections.ConfigDict()
    config.dataset_batch_size = 8192
    config.dataset_sequence_length = 256
    config.learning_rate = 0.01
    config.momentum = 0.9
    config.model_embed_dim = 4
    config.transformer_num_layers = 2
    config.transformer_num_attention_heads = 2
    config.transformer_attention_size_per_head = 2
    config.transformer_dropout_rate = 0.1
    return ml_collections.FrozenConfigDict(config)
