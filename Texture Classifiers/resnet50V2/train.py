from callbacks import get_call_backs
from resnet50V2.model import get_model
from augmentation import DataGenerator
from tensorflow.keras import optimizers

# parameters
train = True
res_path = 'resnet50_v2_fine.h5'
weight_path = r'resnet50_v2.h5'
num_classes = 7
reduce_lr, tb_callback, \
checkpointer, early_stopping, \
csv_logger = get_call_backs(res_path,
                            reduce_lr_p=2,
                            early_stopping_p=10)
lr = 1e-4
size = (512, 512)
batch_size = 16
epochs = 300
augment = True

model = get_model(num_classes=num_classes,
                  dim=[*size, 3],
                  weights=weight_path,
                  fine_tune=True)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizers.Adam(lr=lr),
              metrics=['accuracy'])

print(model.summary())

train_generator = DataGenerator('top', r'..', batch_size=batch_size, dim=size, augment=augment)
val_generator = DataGenerator('validation/top', r'..', batch_size=batch_size, dim=size, augment=False, val=True)
model.fit(train_generator,
          steps_per_epoch=len(train_generator),
          validation_data=val_generator,
          validation_steps=len(val_generator),
          epochs=epochs,
          verbose=1,
          workers=4,
          callbacks=[reduce_lr, tb_callback, checkpointer, early_stopping, csv_logger])
