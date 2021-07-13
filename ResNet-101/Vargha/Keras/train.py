from callbacks import get_call_backs
from model import get_model
from augmentation import DataGenerator
from tensorflow.keras import optimizers

# parameters
train = True
res_path = r'resnet101.h5'
weight_path = None
num_classes = 7
reduce_lr, tb_callback, checkpointer, early_stopping, csv_logger = get_call_backs(res_path,
                                                                                  reduce_lr_p=10,
                                                                                  early_stopping_p=50)
lr = 1e-3
size = (224, 224)
batch_size = 32
epochs = 300
augment = True

model = get_model(num_classes=num_classes)
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
