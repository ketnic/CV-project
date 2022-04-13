from . import Config


test_config = Config.Config(
    VALIDATION_SPLIT=0.2,
    BALANCE=True,
    BASE_MODEL=Config.Models.resnet101.value,
    DENSE_1=256,
    DENSE_2=None,
    DROPOUT_RATE=0.5,
    ACTIVATION=Config.ActivationFunctions.relu.value,
    LOSS=Config.LossFunctions.categorical_crossentropy.value
)