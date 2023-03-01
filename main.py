import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
from trainers.eri import ERI
from dataloaders.abaw_snippet import ABAWDataModule


def load_callbacks():
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='val_pcc',
        mode='max',
        patience=10,
        min_delta=0.001
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val_pcc',
        filename='best-{epoch:02d}-{val_acc:.3f}',
        save_top_k=1,
        mode='max',
        save_last=True
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks


def main(args):
    pl.seed_everything(args.seed)

    if args.trainer_name == 'eri':
        model = ERI(**vars(args))#.cuda()
        data_module = ABAWDataModule(**vars(args))
    else:
        print('Invalid model')

    if args.checkpoint == 'None':
        args.checkpoint = None

    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.log_name)

    trainer = Trainer(deterministic=True,
                      num_sanity_val_steps=10,
                      resume_from_checkpoint=args.checkpoint,
                      logger=logger,
                      gpus=args.gpus,
                      gradient_clip_val=args.clip_val,
                      max_epochs=args.num_epochs,
                      limit_val_batches=args.limit_val_batches,
                      val_check_interval=args.val_check_interval,
                      accumulate_grad_batches=args.grad_accumulate,
                      fast_dev_run=False,
                      enable_checkpointing=True,
                      callbacks=load_callbacks())

    if args.train == 'True':
        trainer.fit(model, data_module.train_loader, data_module.val_loader)
        trainer.test(model=model, dataloaders=data_module.test_loader)
    else:
        model = model.load_from_checkpoint(args.checkpoint, args=args)
        trainer.test(model=model, dataloaders=data_module.test_loader)


if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('-gpus', default='1', type=str)

    parser.add_argument('-grad_accumulate', type=int, default=1)
    parser.add_argument('-clip_val', default=1.0, type=float)
    parser.add_argument('-limit_val_batches', default=1.0, type=float)
    parser.add_argument('-val_check_interval', default=1.0, type=float)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_steps', default=20, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # Restart Control
    parser.add_argument('--checkpoint', default='None', type=str)

    # Training Info
    parser.add_argument('--train', default='True', type=str)
    parser.add_argument('--dataset_folder_path', default='./dataset/abaw5', type=str)
    parser.add_argument('--input_size', default=299, type=int)
    parser.add_argument('--snippet_size', default=30, type=int)

    parser.add_argument('--trainer_name', default='eri', type=str)
    parser.add_argument('--model_name', default='SMMNet', type=str)
    parser.add_argument('--loss', default='bce', type=str)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='lightning_logs', type=str)
    parser.add_argument('--log_name', default='eri', type=str)
    parser.add_argument('-num_epochs', type=int, default=10)

    # Model Hyperparameters
    # TODO: add

    args = parser.parse_args()
    main(args)
