from torcheval.metrics import (
    # Classification
    BinaryAccuracy,
    BinaryAUPRC,
    BinaryAUROC,
    BinaryBinnedAUPRC,
    BinaryBinnedAUROC,
    BinaryBinnedPrecisionRecallCurve,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryNormalizedEntropy,
    BinaryPrecision,
    BinaryPrecisionRecallCurve,
    BinaryRecall,
    BinaryRecallAtFixedPrecision,

    MulticlassAccuracy,
    MulticlassAUPRC,
    MulticlassAUROC,
    MulticlassBinnedAUPRC,
    MulticlassBinnedAUROC,
    MulticlassBinnedPrecisionRecallCurve,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassPrecisionRecallCurve,
    MulticlassRecall,

    MultilabelAccuracy,
    MultilabelAUPRC,
    MultilabelBinnedAUPRC,
    MultilabelBinnedPrecisionRecallCurve,
    MultilabelPrecisionRecallCurve,
    MultilabelRecallAtFixedPrecision,
    TopKMultilabelAccuracy
)

# All metrics for classification puzzle
available_metrics = {
    "BinaryAccuracy": BinaryAccuracy,
    "BinaryAUPRC": BinaryAUPRC,
    "BinaryAUROC": BinaryAUROC,
    "BinaryBinnedAUPRC": BinaryBinnedAUPRC,
    "BinaryBinnedAUROC": BinaryBinnedAUROC,
    "BinaryBinnedPrecisionRecallCurve": BinaryBinnedPrecisionRecallCurve,
    "BinaryConfusionMatrix": BinaryConfusionMatrix,
    "BinaryF1Score": BinaryF1Score,
    "BinaryNormalizedEntropy": BinaryNormalizedEntropy,
    "BinaryPrecision": BinaryPrecision,
    "BinaryPrecisionRecallCurve": BinaryPrecisionRecallCurve,
    "BinaryRecall": BinaryRecall,
    "BinaryRecallAtFixedPrecision": BinaryRecallAtFixedPrecision,

    "MulticlassAccuracy": MulticlassAccuracy,
    "MulticlassAUPRC": MulticlassAUPRC,
    "MulticlassAUROC": MulticlassAUROC,
    "MulticlassBinnedAUPRC": MulticlassBinnedAUPRC,
    "MulticlassBinnedAUROC": MulticlassBinnedAUROC,
    "MulticlassBinnedPrecisionRecallCurve": MulticlassBinnedPrecisionRecallCurve,
    "MulticlassConfusionMatrix": MulticlassConfusionMatrix,
    "MulticlassF1Score": MulticlassF1Score,
    "MulticlassPrecision": MulticlassPrecision,
    "MulticlassPrecisionRecallCurve": MulticlassPrecisionRecallCurve,
    "MulticlassRecall": MulticlassRecall,

    # Not tested
    "MultilabelAccuracy": MultilabelAccuracy,
    "MultilabelAUPRC": MultilabelAUPRC,
    "MultilabelBinnedAUPRC": MultilabelBinnedAUPRC,
    "MultilabelBinnedPrecisionRecallCurve": MultilabelBinnedPrecisionRecallCurve,
    "MultilabelPrecisionRecallCurve": MultilabelPrecisionRecallCurve,
    "MultilabelRecallAtFixedPrecision": MultilabelRecallAtFixedPrecision,
    "TopKMultilabelAccuracy": TopKMultilabelAccuracy
}
