from callbacks import get_call_backs
from mobilenet_v2.model import get_model
from augmentation import DataGenerator
from tensorflow.keras import optimizers

# parameters
train = True
res_path = r'mobilenet_v2_new_val.h5'
weight_path = None
num_classes = 7
reduce_lr, tb_callback, checkpointer, early_stopping, csv_logger = get_call_backs(res_path,
                                                                                  early_stopping_p=50,
                                                                                  reduce_lr_p=10)
lr = 1e-2
size = (224, 224)
batch_size = 32
epochs = 200
augment = True

model = get_model(num_classes=num_classes)
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
                    epochs=epochs,
                    verbose=1,
                    workers=4,
                    callbacks=[reduce_lr, tb_callback, checkpointer, early_stopping, csv_logger])
