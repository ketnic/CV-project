from . import Config


train_config = Config.Config(
    VALIDATION_SPLIT=0.1,
    BALANCE=True,
    BASE_MODEL=Config.Models.resnet101.value,
    TRAINABLE=True,
    DENSE_1=256,
    DENSE_2=None,
    DROPOUT_RATE=0.5,
    ACTIVATION=Config.ActivationFunctions.relu.value,
    LOSS=Config.LossFunctions.categorical_crossentropy.value,
    EPOCHS=10,
    LOSS_LIMIT_ENABLED=False,
    LOSS_LIMIT=0.00001,
    SAVE_MODEL=False,
    SAVE_BEST_MODEL_ONLY=True,
    SAVE_MODEL_INTO_OUTPUT=False,
    SAVE_METRICS=True,
    CREATE_AND_SAVE_CONFUSION_MATRIX=True,
    SAVE_HISTORY=True
)