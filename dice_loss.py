import torch
import torch.nn as nn


class MultiClassDiceCoeff(nn.Module):
    def __init__(self, num_classes, smooth=1e-6, skip_bg=True):
        super(MultiClassDiceCoeff, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.skip_bg = skip_bg

    def forward(self, inputs, targets):
        '''
        inputs:UNet的输出 (N, C, H, W)
        targets: ground-truth mask (N, H, W)
        '''
        inputs = torch.softmax(inputs, dim=1)

        # (N, H，W)-> (N, C, H, W)
        targets = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        if self.skip_bg:
            inputs = inputs[:, 1:, ...]  # 把channel0去掉
            targets = targets[:, 1:, ...]  # 把channel0去掉

        inputs = inputs.reshape(-1, self.num_classes)
        targets = targets.reshape(-1, self.num_classes)

        intersection = (inputs * targets).sum(dim=0)
        union = inputs.sum(dim=0) + targets.sum(dim=0)
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        return dice_coeff.mean()


class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6, skip_bg=True):
        super(MultiClassDiceLoss, self).__init__()

        self.dice_coeff = MultiClassDiceCoeff(num_classes, smooth, skip_bg)

    def forward(self, inputs, targets):
        dice_coeff = self.dice_coeff(inputs, targets)

        # Return the negative of the mean Dice coefficient (to maximize Dice loss)
        return 1.0 - dice_coeff


# Example usage:
# num_classes = 3 # Number of classes in your dataset
# multi_class_dice_loss = MultiClassDiceLoss(num_classes=num_classes)
# inputs = torch.randn((4, num_classes, 256, 256)) # Example input tensor with batch size of 4 and image size of 256x256
# targets = torch.randint(0, num_classes, (4, 256, 256)) # Example target tensor with integer labels
# loss = multi_class_dice_loss(inputs, targets)
# print(loss)

if __name__ == '__main__':
    preds = torch.randn((20, 2, 128, 128))
    targets = torch.randint(0, 2, (20, 128, 128))
    dice_loss = MultiClassDiceLoss(2)
    print(dice_loss(preds, targets))
