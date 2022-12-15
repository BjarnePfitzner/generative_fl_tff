import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix


def get_cm_image(generated_dataset, classifier, class_labels, file_name=None):
    y_labels = []
    y_true = []
    for batch in generated_dataset.as_numpy_iterator():
        images, labels = batch
        batch_y_pred = classifier.predict(images)
        batch_y_labels = np.argmax(batch_y_pred, axis=1)
        y_labels += batch_y_labels.tolist()
        y_true += labels.tolist()

    cm = confusion_matrix(y_true=y_true, y_pred=y_labels)
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)

    _create_image(cm_df, file_name)

    return cm_df


def _create_image(conf_mat: pd.DataFrame, file_name=None):
    fig = plt.figure(figsize=(len(conf_mat.columns) / 4 * 1.4, len(conf_mat.columns) / 4), dpi=300)
    sn.set(font_scale=0.5)
    sn.heatmap(conf_mat, annot=True, fmt='g')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()

    plt.savefig(file_name, format='png')

    plt.close(fig)
