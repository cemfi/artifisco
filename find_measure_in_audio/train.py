from model import AudioMeasureFinderModel

if __name__ == '__main__':
    from pytorch_lightning import Trainer

    model = AudioMeasureFinderModel(
        root='/mnt/data/datasets/artifisco/data',
        batch_size=5
    )

    # trainer = Trainer(nb_sanity_val_steps=0)
    trainer = Trainer(gpus=-1, nb_sanity_val_steps=0, distributed_backend='ddp', min_nb_epochs=100)
    trainer.fit(model)
