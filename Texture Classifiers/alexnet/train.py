from callbacks import get_call_backs
from alexnet.model import get_model
from augmentation import DataGenerator

# parameters
train = True
res_path = r'alexnet_weight.h5'
weight_path = None
num_classes = 7
reduce_lr, tb_callback, checkpointer, early_stopping = get_call_backs(res_path)
lr = 1e-3
size = (227, 227)
batch_size = 32
epochs = 100

model = get_model(num_classes, lr, weights=weight_path)

print(model.summary())

train_generator = DataGenerator('top', r'..', batch_size=batch_size, dim=size, augment=True)
val_generator = DataGenerator('validation/top', r'..', batch_size=batch_size, dim=size, augment=False, val=True)
model.fit_generator(train_generator,
                    steps_per_epoch=len(train_generator),
                    validation_data=val_generator,
                    validation_steps=len(val_generator),
                    epochs=100,
                    verbose=1,
                    workers=4,
                    callbacks=[reduce_lr, tb_callback, checkpointer, early_stopping])
