from callbacks import get_call_backs
from resnet152.model import get_model
from augmentation import DataGenerator
from tensorflow.keras import optimizers

# parameters
res_path = '/home/symo-pouya/projects/texture/resnet152/resnet152.h5'
fine_path = r'resnet152_fine.h5'
weight_path = None
num_classes = 7
reduce_lr, tb_callback, checkpointer, early_stopping, csv_logger = get_call_backs(fine_path)
lr = 1e-4
size = (224, 224)
batch_size = 32
epochs = 100
augment = True

model = get_model(num_classes=num_classes, weights=res_path, fine_tune=True)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizers.Adam(lr=lr),
              metrics=['accuracy'])

print(model.summary())

train_generator = DataGenerator('top', r'..', batch_size=batch_size, dim=size, augment=augment)
val_generator = DataGenerator('validation/top', r'..', batch_size=batch_size, dim=size, augment=False, val=True)
model.fit_generator(train_generator,
                    steps_per_epoch=len(train_generator),
                    validation_data=val_generator,
                    validation_steps=len(val_generator),
                    epochs=100,
                    verbose=1,
                    workers=4,
                    callbacks=[reduce_lr, tb_callback, checkpointer, early_stopping, csv_logger])
