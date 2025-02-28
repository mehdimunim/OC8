# OC8 


## Architectures à tester


Spécifique segmentation 

- UNet 
- Unet Mini
- VGG16 Unet (segmentation)

==> modifier Loss function 
balance cross entropy
+ autre loss : dice loss 
+ mixte 2 loss : dice loss / balance cross entropy


## Label

numéro de classe <=> groupe 

Mapping entre le numéro de label (1, 2, 3...) et le nom du groupe (flat, human, vehicle....)


Target : labelIds ulm_000000_000019_gtFine_labelIds.png

## Métriques

- Dice coef. (equivalent du F1-score)

- Intersection over union (IoU)

+ autres métriques 


## TODO

==> vérifier le mapping entre les classes

