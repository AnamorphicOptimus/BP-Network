stages = [
    dict(
        epochs=5,  # number of training epochs
        optimizer=dict(type='SGD', lr=1e-2, momentum=0.9, weight_decay=1e-4),  # type of optimizer, learning rate, momentum, weight decay rate
        lr_schedule=dict(type='epoch', policy='step', step=[2, 4]),  # drop the learning rate by 1/10 at epoch 2 and 4
        warmup=dict(type='iter', policy='linear', steps=500, ratio=1 / 3.0),  # learning rate increase linearly from 1e-2/3 to 1e-2 within the fisrt 500 iterations
        validation=dict(interval=1))  # validate every 1 epoch
]

hooks = [
    dict(type='IterTimerHook'),
    dict(type='LrUpdaterHook'),
    dict(type='OptimizerHook'),
    dict(type='CheckpointHook'),  # save checkpoint
    dict(
        type='EventWriterHook',
        interval=50,
        writers=[
            dict(type='CommandLineWriter'),  # print logs in terminal
            dict(type='JSONWriter'),  # write logs into a json file
            dict(type='TensorboardWriter')  # write logs into tensorboard
        ])
]

work_dir = 'work_dirs/mnist'
