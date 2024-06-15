RAZUMEVANJE_2.md

The getattr() function in Python is a built-in function that retrieves the value of a named attribute from an object. If the named attribute does not exist, it raises an AttributeError unless a default value is provided as a third argument. The function is useful for accessing attributes dynamically when the attribute names are not known until runtime 124.

Here's the basic syntax for getattr():

getattr(object, name[, default])

    object: The object from which you want to retrieve the attribute.
    name: A string representing the name of the attribute to retrieve.
    default: Optional. A value to return if the specified attribute does not exist.



Zakaj tf se v utils.py v ResourceManager v trace_layer() ta h pa w outputa upoštevata?
h = y.shape[2]
w = y.shape[3]
self.cur_flops += h * w * layer.weight.size(0) * (layer.weight.size(1) - n_removed_filters) * layer.weight.size(2) * layer.weight.size(3)





model.modules():

import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down_conv_1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.down_conv_2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # More layers would be added here

    def forward(self, x):
        # Define the forward pass
        x = self.down_conv_1(x)
        x = self.down_conv_2(x)
        return x

model = UNet()
if 'unet' in 'unet_model_name':
    for module in model.modules():
        print(module)

Prints:
UNet(
  (down_conv_1): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (down_conv_2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
)
Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

Zato pa ima:
if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Dropout, nn.Upsample, nn.ConvTranspose2d)):

Da samo te gleda. Ne vem pa zakaj gleda:
y = layer.old_forward(x)
če ni Conv2d

Pri densenet pa pač gre po otrocih dokler se ne zabije v basic layer:
The model.children() method in PyTorch is used to directly access the immediate child modules of a model. Unlike model.modules(), which iterates over all modules in the model, including the model itself and all nested modules, model.children() only returns the direct children of the model, not including the model itself or any deeper nested modules.

Pomoje je edino smiselno z .modules() in gledat samo za te layerje, ki jih mi actually uporabljamo.
Mogoče tudi možnost, da oseba doda dict z "ime_layerja" : lambda_ki_kbi_se_izvedla





Parametri v trace_layer se računajo takole:
y = layer.old_forward(x)
h = y.shape[2]
w = y.shape[3]
self.cur_flops += h * w * layer.weight.size(0) * (layer.weight.size(1) - n_removed_filters) * layer.weight.size(2) * layer.weight.size(3)

Ker konvolucija tako deluje. Za vsak output pixel izvede toliko operacij. Ker te weights so pač za en kernel.



V ResourceManager literally nastavimo flops na 0 in pa spremenimo forward vseh metod, da sploh ni forward, ampak prišteva številu FLOPOV:
(temu se reče monkey patching)
self.cur_flops = 0

self.cur_flops += h * w * layer.weight.size(0) * (layer.weight.size(1) - n_removed_filters) * layer.weight.size(2) * layer.weight.size(3)

Na prvo žogo bi raje delal deepcopy, čeprav je tule res kinda nice, da ni treba nič kopirat layerjev.
Samo prevežemo na old_forward, naredimo bogus forward, in potem prevežemo nazaj.
JE actually dokaj nice ja.











poglej kaj so argumenti v start prunning. Pač literally primer teh argumentov.
Potem pa kaj so v start prunning te ki grejo v ta pruna:and:retrain al kaj je.
Da bom imel točno predstavo, kaj uporabnik podaja.
Ker ozadje mi je zdaj kinda jasno.



- args: pomoje dict
- device: torch.device("cuda")
- start_epoch = 0 #parameters['start_epoch'] #args.startEpoch
- end_epoch = start_epoch + parameters['n_epochs_for_retraining'] #args.epochs
- viz: = Visdom(port=DEFAULT_PORT, server=DEFAULT_HOSTNAME)

- win_train_mIoU_during_retraining: = viz.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(
            title='Train mIoU during retraining'
        )
    )
- win_train_loss_during_retraining neki podobnega

- model: model_dict['unet4'] = UNet(n_classes=4, pretrained=True) iz models.py
- optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


- criterion: = CrossEntropyLoss2d()
- criterion_DICE: = GeneralizedDiceLoss(softmax=True, reduction=True, n_classes=4 if 'sip' in args.dataset.lower() else 2)
- criterion_SL: = SurfaceLoss()


    # PARAMETERS:
    parameters = {
        'omega': omega,  # importance = ω * weight criterion + (1 - ω) * activation criterion
        #'learnable_parameters_min_limit': 74788,  # compressed student has this much parameters
        #'flops_min_limit': 0, # end when min flops are reached
        'prune_away_percent': 74, # How many percentage of constraints should be pruned away. E.g., 50 means 50% of FLOPs will be pruned away
        #'n_filters_to_remove_at_once': 10,
        'n_removed_flops_for_this_step': 1000000000,
        #'start_epoch': 0,
        'n_epochs_for_retraining': 5,
        'block_limit_in_percents': 75,  # how many parameters can be disabled in one block
        'layer_limit_in_percents': 75,  # how many parameters can be disabled in one layer
        'alpha': None,  # will take 0 as value
        'alpha_original': 1,
        # -------------------HINTON-------------------
        'alpha_distillation': dalpha,
        'T': dtemp,
        # ---------------ATTENTION------------------
        'beta': dbeta, # beta 0.25 in hinton 0.00005 sta priblizno 30 oba. in original je 30. Ideja je, da hoces vec destilacije kot originalnega lossa
        # -------------FSP------------------
        'lambda': dlambda
    }



A imamo to samo zato, da lahko preverimo, koliko je bilo na začetku parametrov v vsakem tem?
JA TO SAMO ZATO. Včasih se je še v get_zeroed_and_all_parameters_wrt_layer_for_model_and_zeroed_percent_for_layer_and_block nekaj ponastavljajlo, ampak se zdaj ne več.

To bi potem bilo najbolje kar naredit programmably.
name := 'down1.conv'
layer = op.attrgetter(name)(model)
Potem samo: count_learnable_parameters_for_module ki dela samo če imaš single convolutional filter (slabo ime imho)
Ta potem uporabi count_zeroed_filter_for_layer da dobi koliko filtrov ni zeroed out (ker drugi način da onemogočimo nek filter je, da ga damo vsega na 0 pa pomoje izklopimo grad nanjem al neki)
In potem n_learnable_weights = module.kernel_size[0] * module.kernel_size[1] * module.in_channels * len(used_filters).
Ampak, a ni to potem narobej? KEr ne upošteva outputa? Aja, huh, pravilno je. Ker tu dobesedno vse weighte prešteješ.
V ResourceManagerju smo pa gledali koliko flopsov se izvede, ne koliko je uteži, zato smo gledali tudi output piksle.
IN TO JE RAZLIKA MED TEMA METODAMA!!!!! OPAZI ME!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    elif 'unet' in model_name:
        all_parameters_for_block_dict = {
            'inc': 37568,
            'down1.conv':  221440,
            'down2.conv':  885248,
            'down3.conv':  3539968,
            'down4.conv':  4719616,
            'up1.conv': 5899008,
            'up2.conv': 1474944,
            'up3.conv': 368832,
            'up4.conv': 110720,
        }

        all_parameters_layer_name_dict = {
            'inc.conv1': 640,
            'inc.conv2': 36928,
            'down1.conv.conv1': 73856,
            'down1.conv.conv2': 147584,
            'down2.conv.conv1': 295168,
            'down2.conv.conv2': 590080,
            'down3.conv.conv1': 1180160,
            'down3.conv.conv2': 2359808,
            'down4.conv.conv1': 2359808,
            'down4.conv.conv2': 2359808,
            'up1.conv.conv1': 4719104,
            'up1.conv.conv2': 1179904,
            'up2.conv.conv1': 1179904,
            'up2.conv.conv2': 295040,
            'up3.conv.conv1': 295040,
            'up3.conv.conv2': 73792,
            'up4.conv.conv1': 73792,
            'up4.conv.conv2': 36928
        }



- teacher_model_copy = copy.deepcopy(teacher_model) # check that teacher is not changing
