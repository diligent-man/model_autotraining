import torch
from torcheval.metrics import MulticlassF1Score, MulticlassPrecisionRecallCurve, MulticlassConfusionMatrix, MulticlassBinnedPrecisionRecallCurve
from typing import List
from multipledispatch import dispatch
from matplotlib import pyplot as plt

plt.switch_backend("tkagg")


@dispatch(torch.Tensor)
def get_metric_result(computed_metric: torch.Tensor) -> List:
    return computed_metric.item() if computed_metric.dim() == 1 and len(computed_metric) == 1 else computed_metric.detach().cpu().numpy().tolist()


@dispatch(tuple)
def get_metric_result(computed_metric: tuple) -> None:
    metric_result = []
    for constituent_metric in computed_metric:
        contituent_result = []
        for tensor in constituent_metric:
            contituent_result.append(get_metric_result(tensor))
        metric_result.append(contituent_result)
    return metric_result


def main() -> None:
    device = "cuda"

    # (num_samples, num_classes) (2, 6)
    input = torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5], [0.7, 0.7, 0.7, 0.7], [0.8, 0.8, 0.8, 0.8]], device=device)
    # (num_samples,) (2,)
    target = torch.tensor([0, 1, 2, 3], device=device)

    num_classes = input.shape[1]
    threshold = torch.arange(0, 1, 0.1).tolist()
    metric_names = ["f1", "prc", "conf", "binprc"]

    metrics = [MulticlassF1Score(num_classes=num_classes, device=device),
               MulticlassPrecisionRecallCurve(num_classes=num_classes, device=device),
               MulticlassConfusionMatrix(num_classes=num_classes, device=device),
               MulticlassBinnedPrecisionRecallCurve(num_classes=num_classes, threshold=threshold, device=device)
               ]

    metrics = [metric.update(input, target).compute() for metric in metrics]
    metrics = [get_metric_result(metric) for metric, metric_name in zip(metrics, metric_names)]

    for metric in metrics:
        print(metric)

    # calculate precision and recall
    precision, recall, thresholds = metrics[1]

    # create precision recall curve
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='purple')

    # add axis labels to plot
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

    # display plot
    plt.show()

    return None


if __name__ == '__main__':
    main()
