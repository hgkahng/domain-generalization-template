
import os
import sys
sys.path.insert(
    0, 
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

import argparse

import torch
import torch.nn as nn

from dg_template.datasets.pacs import PACSDataModule
from dg_template.transforms.domainbed import DomainBedTransform
from dg_template.networks.resnet import ResNetBackbone


def parse_arguments():

    parser = argparse.ArgumentParser(description="ERM on PACS dataset.", add_help=True)
    
    parser.add_argument('--max_iters', type=int, default=5000, help='Number of iterations to train.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay.')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    
    parser.add_argument('--data_dir', type=str, default='data/domainbed/pacs', help='Data directory.')
    
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loader.')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of classes.')
    
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model.')
    
    parser.add_argument('--eval', action='store_true', help='Evaluate model.')
    parser.add_argument('--eval_freq', type=int, default=0, help='Step to evaluate model from.')

    return parser.parse_args()


def main(args: argparse.Namespace):

    # Device configuration
    if not torch.cuda.is_available():
        setattr(args, 'device', 'cpu')
    else:
        setattr(args, 'device', f'cuda:{args.gpu}')

    # Datasets
    pacs_dm = PACSDataModule(
        root=args.data_dir,
        train_environments=['P', 'A', 'C'],
        test_environments=['S'],
        validation_size=0.2,
        random_state=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Data loaders
    train_loader = pacs_dm.train_dataloader(infinite=True)
    eval_loader = pacs_dm.val_dataloader()
    test_loader = pacs_dm.test_dataloader()

    # Transforms
    train_transform = DomainBedTransform(augmentation=True)
    eval_transform = DomainBedTransform(augmentation=False)
    train_transform = train_transform.to(args.device)
    eval_transform = eval_transform.to(args.device)

    # Model
    backbone = ResNetBackbone('resnet18', pretrained=args.pretrained)
    classifier = nn.Linear(backbone.out_features, args.num_classes)
    backbone = backbone.to(args.device)
    classifier = classifier.to(args.device)

    # Optimizer
    optimizer = torch.optim.Adam(
        params=[
            {'params': backbone.parameters()},
            {'params': classifier.parameters()},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Train
    backbone.train()
    classifier.train()

    for i, batch in enumerate(train_loader):

        x = batch['x'].to(args.device)
        y = batch['y'].to(args.device)
        x = train_transform(x)
        y = y.long()

        optimizer.zero_grad()
        logits = classifier(backbone(x))
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        optimizer.step()

        # Eval
        if i % args.eval_freq == 0:

            backbone.eval()
            classifier.eval()

            with torch.no_grad():

                for batch in eval_loader:

                    x = batch['x'].to(args.device)
                    y = batch['y'].to(args.device)
                    x = eval_transform(x)
                    y = y.long()

                    logits = classifier(backbone(x))
                    loss = nn.CrossEntropyLoss()(logits, y)

                    print(f'iter: {i} | loss: {loss:.4f}')
            
            backbone.train()
            classifier.train()

        
if __name__ == '__main__':

    args = parse_arguments()
    main(args);
