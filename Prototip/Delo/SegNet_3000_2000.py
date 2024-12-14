import torch
import torch.nn as nn
import torch.nn.functional as F

from padding import pad_or_resize_to_dims



class SegNet(nn.Module):

    def __init__(self, output_y, output_x, in_chn=3, out_chn=32, BN_momentum=0.5, expansion=2, starting_kernels=64):
        super(SegNet, self).__init__()

        self.output_y = output_y 
        self.output_x = output_x

        #SegNet Architecture
        #Takes input of size in_chn = 3 (RGB images have 3 channels)
        #Outputs size label_chn (N # of classes)

        #ENCODING consists of 5 stages
        #Stage 1, 2 has 2 layers of Convolution + Batch Normalization + Max Pool respectively
        #Stage 3, 4, 5 has 3 layers of Convolution + Batch Normalization + Max Pool respectively

        #General Max Pool 2D for ENCODING layers
        #Pooling indices are stored for Upsampling in DECODING layers

        

        # This is the most awesome way to do it. The most clear, simple, and not prone to mistakes.
        # Especially if you laready have it done for expansion factor == 2.
        # Then you can just search and replace 64 with kernels[0], 128 with kernels[1], etc.
        kernels = []
        for i in range(4):
            kernels.append(int(starting_kernels * (expansion**i)))



        self.ConvEn11 = nn.Conv2d(in_chn, kernels[0], kernel_size=3, padding=1)
        self.BNEn11 = nn.BatchNorm2d(kernels[0], momentum=BN_momentum)


        self.ConvEn12 = nn.Conv2d(kernels[0], kernels[0], kernel_size=3, padding=1)
        self.BNEn12 = nn.BatchNorm2d(kernels[0], momentum=BN_momentum)
        self.MaxEn1 = nn.MaxPool2d(2, stride=2, return_indices=True)

        
        self.ConvEn21 = nn.Conv2d(kernels[0], kernels[1], kernel_size=3, padding=1)
        self.BNEn21 = nn.BatchNorm2d(kernels[1], momentum=BN_momentum)
        self.ConvEn22 = nn.Conv2d(kernels[1], kernels[1], kernel_size=3, padding=1)
        self.BNEn22 = nn.BatchNorm2d(kernels[1], momentum=BN_momentum)
        self.MaxEn2 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.ConvEn31 = nn.Conv2d(kernels[1], kernels[2], kernel_size=3, padding=1)
        self.BNEn31 = nn.BatchNorm2d(kernels[2], momentum=BN_momentum)
        self.ConvEn32 = nn.Conv2d(kernels[2], kernels[2], kernel_size=3, padding=1)
        self.BNEn32 = nn.BatchNorm2d(kernels[2], momentum=BN_momentum)
        self.ConvEn33 = nn.Conv2d(kernels[2], kernels[2], kernel_size=3, padding=1)
        self.BNEn33 = nn.BatchNorm2d(kernels[2], momentum=BN_momentum)
        self.MaxEn3 = nn.MaxPool2d(2, stride=2, return_indices=True)


        self.ConvEn41 = nn.Conv2d(kernels[2], kernels[3], kernel_size=3, padding=1)
        self.BNEn41 = nn.BatchNorm2d(kernels[3], momentum=BN_momentum)
        self.ConvEn42 = nn.Conv2d(kernels[3], kernels[3], kernel_size=3, padding=1)
        self.BNEn42 = nn.BatchNorm2d(kernels[3], momentum=BN_momentum)
        self.ConvEn43 = nn.Conv2d(kernels[3], kernels[3], kernel_size=3, padding=1)
        self.BNEn43 = nn.BatchNorm2d(kernels[3], momentum=BN_momentum)
        self.MaxEn4 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.ConvEn51 = nn.Conv2d(kernels[3], kernels[3], kernel_size=3, padding=1)
        self.BNEn51 = nn.BatchNorm2d(kernels[3], momentum=BN_momentum)
        self.ConvEn52 = nn.Conv2d(kernels[3], kernels[3], kernel_size=3, padding=1)
        self.BNEn52 = nn.BatchNorm2d(kernels[3], momentum=BN_momentum)
        self.ConvEn53 = nn.Conv2d(kernels[3], kernels[3], kernel_size=3, padding=1)
        self.BNEn53 = nn.BatchNorm2d(kernels[3], momentum=BN_momentum)
        # self.MaxEn5 = nn.MaxPool2d(2, stride=2, return_indices=True)

        # Commented out MaxEn5 and MaxDe1.
        # Why would we pool and then immediately unpool?
        # We just loose info fo no reason.

        #DECODING consists of 5 stages
        #Each stage corresponds to their respective counterparts in ENCODING

        #General Max Pool 2D/Upsampling for DECODING layers

        # self.MaxDe1 = nn.MaxUnpool2d(2, stride=2)
        self.ConvDe53 = nn.Conv2d(kernels[3], kernels[3], kernel_size=3, padding=1)
        self.BNDe53 = nn.BatchNorm2d(kernels[3], momentum=BN_momentum)
        self.ConvDe52 = nn.Conv2d(kernels[3], kernels[3], kernel_size=3, padding=1)
        self.BNDe52 = nn.BatchNorm2d(kernels[3], momentum=BN_momentum)
        self.ConvDe51 = nn.Conv2d(kernels[3], kernels[3], kernel_size=3, padding=1)
        self.BNDe51 = nn.BatchNorm2d(kernels[3], momentum=BN_momentum)

        self.MaxDe2 = nn.MaxUnpool2d(2, stride=2)
        self.ConvDe43 = nn.Conv2d(kernels[3], kernels[3], kernel_size=3, padding=1)
        self.BNDe43 = nn.BatchNorm2d(kernels[3], momentum=BN_momentum)
        self.ConvDe42 = nn.Conv2d(kernels[3], kernels[3], kernel_size=3, padding=1)
        self.BNDe42 = nn.BatchNorm2d(kernels[3], momentum=BN_momentum)
        self.ConvDe41 = nn.Conv2d(kernels[3], kernels[2], kernel_size=3, padding=1)
        self.BNDe41 = nn.BatchNorm2d(kernels[2], momentum=BN_momentum)

        self.MaxDe3 = nn.MaxUnpool2d(2, stride=2)
        self.ConvDe33 = nn.Conv2d(kernels[2], kernels[2], kernel_size=3, padding=1)
        self.BNDe33 = nn.BatchNorm2d(kernels[2], momentum=BN_momentum)
        self.ConvDe32 = nn.Conv2d(kernels[2], kernels[2], kernel_size=3, padding=1)
        self.BNDe32 = nn.BatchNorm2d(kernels[2], momentum=BN_momentum)
        self.ConvDe31 = nn.Conv2d(kernels[2], kernels[1], kernel_size=3, padding=1)
        self.BNDe31 = nn.BatchNorm2d(kernels[1], momentum=BN_momentum)

        self.MaxDe4 = nn.MaxUnpool2d(2, stride=2)
        self.ConvDe22 = nn.Conv2d(kernels[1], kernels[1], kernel_size=3, padding=1)
        self.BNDe22 = nn.BatchNorm2d(kernels[1], momentum=BN_momentum)
        self.ConvDe21 = nn.Conv2d(kernels[1], kernels[0], kernel_size=3, padding=1)
        self.BNDe21 = nn.BatchNorm2d(kernels[0], momentum=BN_momentum)

        self.MaxDe5 = nn.MaxUnpool2d(2, stride=2)
        self.ConvDe12 = nn.Conv2d(kernels[0], kernels[0], kernel_size=3, padding=1)
        self.BNDe12 = nn.BatchNorm2d(kernels[0], momentum=BN_momentum)

        self.ConvDe11 = nn.Conv2d(kernels[0], out_chn, kernel_size=3, padding=1)
        # self.BNDe11 = nn.BatchNorm2d(self.out_chn, momentum=BN_momentum)

    def forward(self, x):

        #ENCODE LAYERS
        #Stage 1
        x = F.relu(self.BNEn11(self.ConvEn11(x))) 
        x = F.relu(self.BNEn12(self.ConvEn12(x))) 
        size_0 = x.size()
        x, ind1 = self.MaxEn1(x)
        size1 = x.size()

        #Stage 2
        x = F.relu(self.BNEn21(self.ConvEn21(x))) 
        x = F.relu(self.BNEn22(self.ConvEn22(x))) 
        x, ind2 = self.MaxEn2(x)
        size2 = x.size()

        #Stage 3
        x = F.relu(self.BNEn31(self.ConvEn31(x))) 
        x = F.relu(self.BNEn32(self.ConvEn32(x))) 
        x = F.relu(self.BNEn33(self.ConvEn33(x)))   
        x, ind3 = self.MaxEn3(x)
        size3 = x.size()

        #Stage 4
        x = F.relu(self.BNEn41(self.ConvEn41(x))) 
        x = F.relu(self.BNEn42(self.ConvEn42(x))) 
        x = F.relu(self.BNEn43(self.ConvEn43(x)))   
        x, ind4 = self.MaxEn4(x)
        size4 = x.size()

        #Stage 5
        x = F.relu(self.BNEn51(self.ConvEn51(x))) 
        x = F.relu(self.BNEn52(self.ConvEn52(x))) 
        x = F.relu(self.BNEn53(self.ConvEn53(x)))

        # Why the hell would we go losing this information here?
        # Why pool and then unpool?

        # x, ind5 = self.MaxEn5(x)
        # # size5 = x.size()

        #DECODE LAYERS
        #Stage 5
        # x = self.MaxDe1(x, ind5, output_size=size4)
        
        x = F.relu(self.BNDe53(self.ConvDe53(x)))
        x = F.relu(self.BNDe52(self.ConvDe52(x)))
        x = F.relu(self.BNDe51(self.ConvDe51(x)))

        #Stage 4
        x = self.MaxDe2(x, ind4, output_size=size3)
        x = F.relu(self.BNDe43(self.ConvDe43(x)))
        x = F.relu(self.BNDe42(self.ConvDe42(x)))
        x = F.relu(self.BNDe41(self.ConvDe41(x)))

        #Stage 3
        x = self.MaxDe3(x, ind3, output_size=size2)
        x = F.relu(self.BNDe33(self.ConvDe33(x)))
        x = F.relu(self.BNDe32(self.ConvDe32(x)))
        x = F.relu(self.BNDe31(self.ConvDe31(x)))

        #Stage 2
        x = self.MaxDe4(x, ind2, output_size=size1)
        x = F.relu(self.BNDe22(self.ConvDe22(x)))
        x = F.relu(self.BNDe21(self.ConvDe21(x)))

        #Stage 1
        x = self.MaxDe5(x, ind1, output_size=size_0)
        x = F.relu(self.BNDe12(self.ConvDe12(x)))

        x = pad_or_resize_to_dims(x, self.output_y, self.output_x)
        x = self.ConvDe11(x)

        # This is wrong! Loss functions expect logits (unnormalized scores) and they do the softmax internally.
        # Doing softmax twice introduces numerical instability and a bunc of problems.
        # x = F.softmax(x, dim=1)

        return x