# https://julienbeaulieu.github.io/2020/03/16/building-a-flexible-configuration-system-for-deep-learning-models/

from trainer import Trainer
from opption_setter import OptionSetter
from data_preprocessor import DataPreprocessor


def main() -> None:
    optionSetter = OptionSetter()

    dataProcessor = DataPreprocessor(path_options=optionSetter.getPathSetup(), dataset_options=optionSetter.getDatasetSetup(),
                                     class_labels=["1"], multipliers=[2], thresholds=[.3]
                                     )
    # dataProcessor.preprocessing()
    train_set, test_set = dataProcessor.split_train_test()
    # train_set = dataProcessor.imgAugmentation(dataset=train_set)
    train_set, test_set = dataProcessor.getDataLoader(train_set, test_set)


    trainer = Trainer(path_options=optionSetter.getPathSetup(),
                      dataset_options=optionSetter.getDatasetSetup(),
                      checkpoint_options=optionSetter.getCheckpointSetup(),
                      optimizer_options=optionSetter.getOptimizerSetup(),
                      metric_options=optionSetter.getMetricSetup(),
                      device_options=optionSetter.getDeviceSetup(),
                      nn_options=optionSetter.getNnSetup(),
                      epoch_options=optionSetter.getEpochSetup()
                      )
    trainer.train(train_set=train_set, test_set=test_set)
    """
    Task lists:
        + Logging with json format (Infinitt & NaN can not read when decoding in KNIME)
        + Add tensor board while training
        + Revise computation of training metrics
        + Take latest model when load_checkpoint == True
        + Save 4 best + 1 latest model
    """
    return None


if __name__ == '__main__':
    main()
