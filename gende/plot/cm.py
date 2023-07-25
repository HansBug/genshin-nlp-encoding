import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

try:
    from typing import Literal, List
except (ImportError, ModuleNotFoundError):
    from typing_extensions import Literal


def cm_func(y_true, y_pred, labels: List[str]):
    N = len(labels)
    arr = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            arr[i, j] = ((y_true == i) & (y_pred == j)).sum()

    return arr


def plt_confusion_matrix(ax, y_true, y_pred, labels, title: str = 'Confusion Matrix',
                         normalize: Literal['true', 'pred', None] = None, cmap=None) -> Image.Image:
    cm = cm_func(y_true, y_pred, labels)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )
    disp.plot(ax=ax, cmap=cmap or plt.cm.Blues)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90)
    ax.set_title(title)
