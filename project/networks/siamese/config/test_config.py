from . import Config


test_config = Config.Config(
    BASE_MODEL=Config.Models.resnet101.value,
    DENSE_1=512,
    DENSE_2=256,
    DROPOUT_RATE=None,
    BATCH_NORM=True,
)
