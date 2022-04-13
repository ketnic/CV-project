from . import Config


train_config = Config.Config(
    BALANCED=True,
    MAKE_ITSELF_PAIRS=True,
    BASE_MODEL=Config.Models.resnet101.value,
    DENSE_1=512,
    DENSE_2=256,
    DROPOUT_RATE=None,
    BATCH_NORM=True,
    LOSS=Config.LossFunctions.binary_crossentropy.value,
    EPOCHS=7,
    LOSS_LIMIT_ENABLED=False,
    LOSS_LIMIT=0.1665,
    SAVE_MODEL_WEIGHTS=True,
    SAVE_MODEL_INTO_OUTPUT=False,
    SAVE_METRICS=True,
    CREATE_AND_SAVE_CONFUSION_MATRIX=True,
    SAVE_HISTORY=True,
)
