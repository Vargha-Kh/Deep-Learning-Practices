import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from augmentation import DataGenerator
from resnet101V2.model import get_model

# parameters
train = True
weight_path = r'resnet101V2/resnet101_v2_fine.h5'
num_classes = 7
size = (512, 512)
batch_size = 16
epochs = 300
augment = True

model = get_model(num_classes=num_classes,
                  dim=[*size, 3],
                  weights=weight_path,
                  fine_tune=True,
                  predict=True)

print(model.summary())

train_generator = DataGenerator('top', r'.',
                                batch_size=batch_size,
                                dim=size,
                                augment=False,
                                val=True,
                                shuffle=False)
val_generator = DataGenerator('validation/top', r'.',
                              batch_size=batch_size,
                              dim=size,
                              augment=False,
                              val=True,
                              shuffle=False)


def report(generator):
    Y_pred = model.predict_generator(generator, len(generator) + 1)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    y_true = np.array([generator.class_name_map[l] for l in generator.labels][:len(y_pred)])
    print(confusion_matrix(y_true, y_pred))
    print('Classification Report')
    # target_names = ['Cats', 'Dogs', 'Horse']
    print(classification_report(y_true,
                                y_pred,
                                target_names=list(generator.class_name_map.keys())))


# Confution Matrix and Classification Report
report(train_generator)
report(val_generator)
