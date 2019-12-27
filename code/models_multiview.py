import torch.nn as nn
from torch.autograd import Variable
import torch
import torchvision.models as models

import numpy as np

class FrontaliseModelMultiViewInputBBoxes(nn.Module):
        def __init__(self, input_nc, num_decoders=5, inner_nc=512, num_additional_ids=0, smaller=False):
                super(FrontaliseModelMultiViewInputBBoxes, self).__init__()
                self.encoder = self.generate_encoder_layers(inner_nc)
                self.encoder1 = self.generate_encoder_small_layers(num_additional_ids)
                self.encoder2 = self.generate_encoder_small_layers(num_additional_ids)
                self.encoder3 = self.generate_encoder_small_layers(num_additional_ids)

                self.encoders = nn.Sequential(self.encoder, self.encoder1, self.encoder2, self.encoder3)

                self.decoder1 = self.generate_decoder_small_layers(num_additional_ids)
                self.decoder2 = self.generate_decoder_small_layers(num_additional_ids)
                self.decoder3 = self.generate_decoder_small_layers(num_additional_ids)
                self.decoder4 = self.generate_decoder_layers(inner_nc)
                self.decoder5 = self.generate_decoder_layers(inner_nc)

                self.decoders = nn.Sequential(self.decoder1, self.decoder2, self.decoder3, self.decoder4, self.decoder5)

        def generate_encoder_small_layers(self, num_additional_ids, num_filters=16):
                conv1 = nn.Conv2d(3, num_filters, 4, 2, 1)
                conv2 = nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1)
                conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1)
                conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1)
                conv5 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv6 = nn.Conv2d(num_filters * 8, num_additional_ids, 4, 2, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)

                leaky_relu = nn.LeakyReLU(0.2, True)
                return nn.Sequential(conv1, leaky_relu, conv2, batch_norm2_0, \
                                                          leaky_relu, conv3, batch_norm4_0, leaky_relu, \
                                                          conv4, batch_norm8_0, leaky_relu, conv5, 
                                                          batch_norm8_1, leaky_relu, conv6)

        def generate_encoder_layers(self, num_additional_ids, num_filters=16):
                conv1 = nn.Conv2d(3, num_filters, 4, 2, 1)
                conv2 = nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1)
                conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1)
                conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1)
                conv5 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv6 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv7 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv8 = nn.Conv2d(num_filters * 8, num_additional_ids, 4, 2, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)

                leaky_relu = nn.LeakyReLU(0.2, True)
                return nn.Sequential(conv1, leaky_relu, conv2, batch_norm2_0, \
                                                          leaky_relu, conv3, batch_norm4_0, leaky_relu, \
                                                          conv4, batch_norm8_0, leaky_relu, conv5, 
                                                          batch_norm8_1, leaky_relu, conv6, batch_norm8_2, 
                                                          leaky_relu, conv7, batch_norm8_3, leaky_relu, conv8)

        def generate_decoder_small_layers(self, num_input_channels, num_output_channels=2,num_filters=16):
                up = nn.Upsample(scale_factor=2, mode='bilinear')

                dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
                dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
                dconv5 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
                dconv6 = nn.Conv2d(num_filters * 2 , num_output_channels, 3, 1, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm4_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_7 = nn.BatchNorm2d(num_filters * 4)

                leaky_relu = nn.LeakyReLU(0.2)
                relu = nn.ReLU()
                tanh = nn.Tanh()

                return nn.Sequential(relu, up, dconv1, batch_norm8_4, \
                                                        relu, up, dconv2, batch_norm8_5, relu,
                                                        up, dconv3, batch_norm8_6, relu, up, dconv4,
                                                        batch_norm8_7, relu, up, dconv5, batch_norm4_1, 
                                                        relu, up, dconv6, tanh)

        def generate_decoder_layers(self, num_input_channels, num_filters=16):
                up = nn.Upsample(scale_factor=2, mode='bilinear')

                dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
                dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 8, 3, 1, 1)
                dconv5 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
                dconv6 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
                dconv7 = nn.Conv2d(num_filters * 2 , num_filters, 3, 1, 1)
                dconv8 = nn.Conv2d(num_filters , 2, 3, 1, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm4_1 = nn.BatchNorm2d(num_filters * 4)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_7 = nn.BatchNorm2d(num_filters * 8)

                leaky_relu = nn.LeakyReLU(0.2)
                relu = nn.ReLU()
                tanh = nn.Tanh()

                return nn.Sequential(relu, up, dconv1, batch_norm8_4, \
                                                        relu, up, dconv2, batch_norm8_5, relu,
                                                        up, dconv3, batch_norm8_6, relu, up, dconv4,
                                                        batch_norm8_7, relu, up, dconv5, batch_norm4_1, 
                                                        relu, up, dconv6, batch_norm2_1, relu, up, dconv7, batch_norm,
                                                        relu, up, dconv8, tanh)


class FrontaliseModelMultiView(nn.Module):
        def __init__(self, input_nc, num_decoders=5, inner_nc=512, num_additional_ids=0, smaller=False, use_mask=False):
                super(FrontaliseModelMultiView, self).__init__()
                self.encoder = self.generate_encoder_layers()
                if smaller:
                        self.decoder1 = self.generate_decoder_small_layers(2+num_additional_ids)
                        self.decoder2 = self.generate_decoder_small_layers(2+num_additional_ids)
                        self.decoder3 = self.generate_decoder_small_layers(2+num_additional_ids)
                else:
                        self.decoder1 = self.generate_decoder_layers(2+num_additional_ids)
                        self.decoder2 = self.generate_decoder_layers(2+num_additional_ids)
                        self.decoder3 = self.generate_decoder_layers(2+num_additional_ids)
                        if use_mask:
                                self.mask_decoder1 = self.generate_decoder_layers(2+num_additional_ids, num_output_channels=1)
                                self.mask_decoder2 = self.generate_decoder_layers(2+num_additional_ids, num_output_channels=1)
                                self.mask_decoder3 = self.generate_decoder_layers(2+num_additional_ids, num_output_channels=1)
                self.decoder4 = self.generate_decoder_layers(4+num_additional_ids)
                self.decoder5 = self.generate_decoder_layers(4+num_additional_ids)
                if use_mask:
                        self.mask_decoder4 = self.generate_decoder_layers(4+num_additional_ids, num_output_channels=1)
                        self.mask_decoder5 = self.generate_decoder_layers(4+num_additional_ids, num_output_channels=1)

                self.decoders = nn.Sequential(self.decoder1, self.decoder2, self.decoder3, self.decoder4, self.decoder5)
                if use_mask:
                        self.mask_decoders = nn.Sequential(self.mask_decoder1, self.mask_decoder2, self.mask_decoder3, self.mask_decoder4, self.mask_decoder5)


        def generate_encoder_layers(self, num_filters=16):
                conv1 = nn.Conv2d(3, num_filters, 4, 2, 1)
                conv2 = nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1)
                conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1)
                conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1)
                conv5 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv6 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv7 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv8 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)

                leaky_relu = nn.LeakyReLU(0.2, True)
                return nn.Sequential(conv1, leaky_relu, conv2, batch_norm2_0, \
                                                          leaky_relu, conv3, batch_norm4_0, leaky_relu, \
                                                          conv4, batch_norm8_0, leaky_relu, conv5, 
                                                          batch_norm8_1, leaky_relu, conv6, batch_norm8_2, 
                                                          leaky_relu, conv7, batch_norm8_3, leaky_relu, conv8)

        def generate_decoder_small_layers(self, num_input_channels, num_output_channels=2, num_filters=16):
                up = nn.Upsample(scale_factor=2, mode='bilinear')

                dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
                dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
                dconv5 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
                dconv6 = nn.Conv2d(num_filters * 2 , num_output_channels, 3, 1, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm4_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_7 = nn.BatchNorm2d(num_filters * 4)

                leaky_relu = nn.LeakyReLU(0.2)
                relu = nn.ReLU()
                tanh = nn.Tanh()

                return nn.Sequential(relu, up, dconv1, batch_norm8_4, \
                                                        relu, up, dconv2, batch_norm8_5, relu,
                                                        up, dconv3, batch_norm8_6, relu, up, dconv4,
                                                        batch_norm8_7, relu, up, dconv5, batch_norm4_1, 
                                                        relu, up, dconv6, tanh)

        def generate_decoder_layers(self, num_input_channels, num_output_channels=2, num_filters=16):
                up = nn.Upsample(scale_factor=2, mode='bilinear')

                dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
                dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 8, 3, 1, 1)
                dconv5 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
                dconv6 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
                dconv7 = nn.Conv2d(num_filters * 2 , num_filters, 3, 1, 1)
                dconv8 = nn.Conv2d(num_filters , num_output_channels, 3, 1, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm4_1 = nn.BatchNorm2d(num_filters * 4)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_7 = nn.BatchNorm2d(num_filters * 8)

                leaky_relu = nn.LeakyReLU(0.2)
                relu = nn.ReLU()
                tanh = nn.Tanh()

                return nn.Sequential(relu, up, dconv1, batch_norm8_4, \
                                                        relu, up, dconv2, batch_norm8_5, relu,
                                                        up, dconv3, batch_norm8_6, relu, up, dconv4,
                                                        batch_norm8_7, relu, up, dconv5, batch_norm4_1, 
                                                        relu, up, dconv6, batch_norm2_1, relu, up, dconv7, batch_norm,
                                                        relu, up, dconv8, tanh)

class FrontaliseModelMultiViewMasks(nn.Module):
        def __init__(self, input_nc, num_decoders=5, inner_nc=128, num_additional_ids=0, smaller=False, num_masks=0):
                super(FrontaliseModelMultiViewMasks, self).__init__()
                self.encoder = self.generate_encoder_layers(output_size=inner_nc)
                
                decoders = [0] * num_masks
                masks = [0] * num_masks
                for i in range(0, num_masks):
                        decoders[i] = self.generate_decoder_layers(inner_nc*2)
                        masks[i] = self.generate_decoder_layers(inner_nc*2, num_output_channels=1)
                
                self.decoders = nn.Sequential(*decoders)
                self.mask_decoders = nn.Sequential(*masks)


        def generate_encoder_layers(self, output_size=128, num_filters=16):
                conv1 = nn.Conv2d(3, num_filters, 4, 2, 1)
                conv2 = nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1)
                conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1)
                conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1)
                conv5 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv6 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv7 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv8 = nn.Conv2d(num_filters * 8, output_size, 4, 2, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)

                leaky_relu = nn.LeakyReLU(0.2, True)
                return nn.Sequential(conv1, leaky_relu, conv2, batch_norm2_0, \
                                                          leaky_relu, conv3, batch_norm4_0, leaky_relu, \
                                                          conv4, batch_norm8_0, leaky_relu, conv5, 
                                                          batch_norm8_1, leaky_relu, conv6, batch_norm8_2, 
                                                          leaky_relu, conv7, batch_norm8_3, leaky_relu, conv8)

        def generate_decoder_layers(self, num_input_channels, num_output_channels=2, num_filters=16):
                up = nn.Upsample(scale_factor=2, mode='bilinear')

                dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
                dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 8, 3, 1, 1)
                dconv5 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
                dconv6 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
                dconv7 = nn.Conv2d(num_filters * 2 , num_filters, 3, 1, 1)
                dconv8 = nn.Conv2d(num_filters , num_output_channels, 3, 1, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm4_1 = nn.BatchNorm2d(num_filters * 4)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_7 = nn.BatchNorm2d(num_filters * 8)

                leaky_relu = nn.LeakyReLU(0.2)
                relu = nn.ReLU()
                tanh = nn.Tanh()

                return nn.Sequential(relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv1, batch_norm8_4, \
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv2, batch_norm8_5, relu,
                                                        nn.Upsample(scale_factor=2, mode='bilinear'), dconv3, batch_norm8_6, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv4,
                                                        batch_norm8_7, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv5, batch_norm4_1, 
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv6, batch_norm2_1, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv7, batch_norm,
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv8, tanh)
                


        def generate_decoder_small_layers(self, num_input_channels, num_output_channels=2, num_filters=16):
                up = nn.Upsample(scale_factor=2, mode='bilinear')

                dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
                dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
                dconv5 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
                dconv6 = nn.Conv2d(num_filters * 2 , num_output_channels, 3, 1, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm4_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_7 = nn.BatchNorm2d(num_filters * 4)

                leaky_relu = nn.LeakyReLU(0.2)
                relu = nn.ReLU()
                tanh = nn.Tanh()

                return nn.Sequential(relu, up, dconv1, batch_norm8_4, \
                                                        relu, up, dconv2, batch_norm8_5, relu,
                                                        up, dconv3, batch_norm8_6, relu, up, dconv4,
                                                        batch_norm8_7, relu, up, dconv5, batch_norm4_1, 
                                                        relu, up, dconv6, tanh)

class FrontaliseModelMasks(nn.Module):
        def __init__(self, input_nc, num_decoders=5, inner_nc=128, num_additional_ids=0, smaller=False, num_masks=0):
                super(FrontaliseModelMasks, self).__init__()
                self.encoder = self.generate_encoder_layers(output_size=inner_nc)
                
                self.decoder = self.generate_decoder_layers(inner_nc*2)
                self.mask = self.generate_decoder_layers(inner_nc*2, num_output_channels=1)


        def generate_encoder_layers(self, output_size=128, num_filters=16):
                conv1 = nn.Conv2d(3, num_filters, 4, 2, 1)
                conv2 = nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1)
                conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1)
                conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1)
                conv5 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv6 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv7 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv8 = nn.Conv2d(num_filters * 8, output_size, 4, 2, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)

                leaky_relu = nn.LeakyReLU(0.2, True)
                return nn.Sequential(conv1, leaky_relu, conv2, batch_norm2_0, \
                                                          leaky_relu, conv3, batch_norm4_0, leaky_relu, \
                                                          conv4, batch_norm8_0, leaky_relu, conv5, 
                                                          batch_norm8_1, leaky_relu, conv6, batch_norm8_2, 
                                                          leaky_relu, conv7, batch_norm8_3, leaky_relu, conv8)

        def generate_decoder_layers(self, num_input_channels, num_output_channels=2, num_filters=16):
                up = nn.Upsample(scale_factor=2, mode='bilinear')

                dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
                dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 8, 3, 1, 1)
                dconv5 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
                dconv6 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
                dconv7 = nn.Conv2d(num_filters * 2 , num_filters, 3, 1, 1)
                dconv8 = nn.Conv2d(num_filters , num_output_channels, 3, 1, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm4_1 = nn.BatchNorm2d(num_filters * 4)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_7 = nn.BatchNorm2d(num_filters * 8)

                leaky_relu = nn.LeakyReLU(0.2)
                relu = nn.ReLU()
                tanh = nn.Tanh()

                return nn.Sequential(relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv1, batch_norm8_4, \
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv2, batch_norm8_5, relu,
                                                        nn.Upsample(scale_factor=2, mode='bilinear'), dconv3, batch_norm8_6, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv4,
                                                        batch_norm8_7, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv5, batch_norm4_1, 
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv6, batch_norm2_1, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv7, batch_norm,
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv8, tanh)
                


        def generate_decoder_small_layers(self, num_input_channels, num_output_channels=2, num_filters=16):
                up = nn.Upsample(scale_factor=2, mode='bilinear')

                dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
                dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
                dconv5 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
                dconv6 = nn.Conv2d(num_filters * 2 , num_output_channels, 3, 1, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm4_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_7 = nn.BatchNorm2d(num_filters * 4)

                leaky_relu = nn.LeakyReLU(0.2)
                relu = nn.ReLU()
                tanh = nn.Tanh()

                return nn.Sequential(relu, up, dconv1, batch_norm8_4, \
                                                        relu, up, dconv2, batch_norm8_5, relu,
                                                        up, dconv3, batch_norm8_6, relu, up, dconv4,
                                                        batch_norm8_7, relu, up, dconv5, batch_norm4_1, 
                                                        relu, up, dconv6, tanh)

class FrontaliseModelMasksNoCat(nn.Module):
        def __init__(self, input_nc, num_decoders=5, inner_nc=128, num_additional_ids=0, smaller=False, num_masks=0):
                super(FrontaliseModelMasksNoCat, self).__init__()
                self.encoder = self.generate_encoder_layers(output_size=inner_nc)
                
                self.decoder = self.generate_decoder_layers(inner_nc)
                self.mask = self.generate_decoder_layers(inner_nc, num_output_channels=1)


        def generate_encoder_layers(self, output_size=128, num_filters=16):
                conv1 = nn.Conv2d(3, num_filters, 4, 2, 1)
                conv2 = nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1)
                conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1)
                conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1)
                conv5 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv6 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv7 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv8 = nn.Conv2d(num_filters * 8, output_size, 4, 2, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)

                leaky_relu = nn.LeakyReLU(0.2, True)
                return nn.Sequential(conv1, leaky_relu, conv2, batch_norm2_0, \
                                                          leaky_relu, conv3, batch_norm4_0, leaky_relu, \
                                                          conv4, batch_norm8_0, leaky_relu, conv5, 
                                                          batch_norm8_1, leaky_relu, conv6, batch_norm8_2, 
                                                          leaky_relu, conv7, batch_norm8_3, leaky_relu, conv8)

        def generate_decoder_layers(self, num_input_channels, num_output_channels=2, num_filters=16):
                up = nn.Upsample(scale_factor=2, mode='bilinear')

                dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
                dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 8, 3, 1, 1)
                dconv5 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
                dconv6 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
                dconv7 = nn.Conv2d(num_filters * 2 , num_filters, 3, 1, 1)
                dconv8 = nn.Conv2d(num_filters , num_output_channels, 3, 1, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm4_1 = nn.BatchNorm2d(num_filters * 4)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_7 = nn.BatchNorm2d(num_filters * 8)

                leaky_relu = nn.LeakyReLU(0.2)
                relu = nn.ReLU()
                tanh = nn.Tanh()

                return nn.Sequential(relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv1, batch_norm8_4, \
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv2, batch_norm8_5, relu,
                                                        nn.Upsample(scale_factor=2, mode='bilinear'), dconv3, batch_norm8_6, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv4,
                                                        batch_norm8_7, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv5, batch_norm4_1, 
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv6, batch_norm2_1, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv7, batch_norm,
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv8, tanh)
                


        def generate_decoder_small_layers(self, num_input_channels, num_output_channels=2, num_filters=16):
                up = nn.Upsample(scale_factor=2, mode='bilinear')

                dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
                dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
                dconv5 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
                dconv6 = nn.Conv2d(num_filters * 2 , num_output_channels, 3, 1, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm4_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_7 = nn.BatchNorm2d(num_filters * 4)

                leaky_relu = nn.LeakyReLU(0.2)
                relu = nn.ReLU()
                tanh = nn.Tanh()

                return nn.Sequential(relu, up, dconv1, batch_norm8_4, \
                                                        relu, up, dconv2, batch_norm8_5, relu,
                                                        up, dconv3, batch_norm8_6, relu, up, dconv4,
                                                        batch_norm8_7, relu, up, dconv5, batch_norm4_1, 
                                                        relu, up, dconv6, tanh)



class FrontaliseModelMultiViewMI(nn.Module):
        def __init__(self, input_nc, num_decoders=5, inner_nc=512, num_additional_ids=0, smaller=False, use_mask=False):
                super(FrontaliseModelMultiViewMI, self).__init__()
                self.encoder = self.generate_encoder_layers(inner_nc)
                if smaller:
                        self.decoder1 = self.generate_decoder_small_layers(num_additional_ids)
                        self.decoder2 = self.generate_decoder_small_layers(num_additional_ids)
                        self.decoder3 = self.generate_decoder_small_layers(num_additional_ids)
                else:
                        self.decoder1 = self.generate_decoder_layers(num_additional_ids)
                        self.decoder2 = self.generate_decoder_layers(num_additional_ids)
                        self.decoder3 = self.generate_decoder_layers(num_additional_ids)
                        if use_mask:
                                self.mask_decoder1 = self.generate_decoder_layers(num_additional_ids, num_output_channels=1)
                                self.mask_decoder2 = self.generate_decoder_layers(num_additional_ids, num_output_channels=1)
                                self.mask_decoder3 = self.generate_decoder_layers(num_additional_ids, num_output_channels=1)
                self.decoder4 = self.generate_decoder_layers(num_additional_ids)
                self.decoder5 = self.generate_decoder_layers(num_additional_ids)
                if use_mask:
                        self.mask_decoder4 = self.generate_decoder_layers(num_additional_ids, num_output_channels=1)
                        self.mask_decoder5 = self.generate_decoder_layers(num_additional_ids, num_output_channels=1)

                self.decoders = nn.Sequential(self.decoder1, self.decoder2, self.decoder3, self.decoder4, self.decoder5)
                if use_mask:
                        self.mask_decoders = nn.Sequential(self.mask_decoder1, self.mask_decoder2, self.mask_decoder3, self.mask_decoder4, self.mask_decoder5)


        def generate_encoder_layers(self, inner_nc, num_filters=16):
                conv1 = nn.Conv2d(3, num_filters, 4, 2, 1)
                conv2 = nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1)
                conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1)
                conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1)
                conv5 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv6 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv7 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv8 = nn.Conv2d(num_filters * 8, inner_nc, 4, 2, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)

                leaky_relu = nn.LeakyReLU(0.2, True)
                return nn.Sequential(conv1, leaky_relu, conv2, batch_norm2_0, \
                                                          leaky_relu, conv3, batch_norm4_0, leaky_relu, \
                                                          conv4, batch_norm8_0, leaky_relu, conv5, 
                                                          batch_norm8_1, leaky_relu, conv6, batch_norm8_2, 
                                                          leaky_relu, conv7, batch_norm8_3, leaky_relu, conv8)

        def generate_decoder_small_layers(self, num_input_channels, num_output_channels=3, num_filters=16):
                up = nn.Upsample(scale_factor=2, mode='bilinear')

                dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
                dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
                dconv5 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
                dconv6 = nn.Conv2d(num_filters * 2 , num_output_channels, 3, 1, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm4_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_7 = nn.BatchNorm2d(num_filters * 4)

                leaky_relu = nn.LeakyReLU(0.2)
                relu = nn.ReLU()
                tanh = nn.Sigmoid()

                return nn.Sequential(relu, up, dconv1, batch_norm8_4, \
                                                        relu, up, dconv2, batch_norm8_5, relu,
                                                        up, dconv3, batch_norm8_6, relu, up, dconv4,
                                                        batch_norm8_7, relu, up, dconv5, batch_norm4_1, 
                                                        relu, up, dconv6, tanh)

        def generate_decoder_layers(self, num_input_channels, num_output_channels=3, num_filters=16):
                up = nn.Upsample(scale_factor=2, mode='bilinear')

                dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
                dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 8, 3, 1, 1)
                dconv5 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
                dconv6 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
                dconv7 = nn.Conv2d(num_filters * 2 , num_filters, 3, 1, 1)
                dconv8 = nn.Conv2d(num_filters , num_output_channels, 3, 1, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm4_1 = nn.BatchNorm2d(num_filters * 4)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_7 = nn.BatchNorm2d(num_filters * 8)

                leaky_relu = nn.LeakyReLU(0.2)
                relu = nn.ReLU()
                tanh = nn.Sigmoid()

                return nn.Sequential(relu, up, dconv1, batch_norm8_4, \
                                                        relu, up, dconv2, batch_norm8_5, relu,
                                                        up, dconv3, batch_norm8_6, relu, up, dconv4,
                                                        batch_norm8_7, relu, up, dconv5, batch_norm4_1, 
                                                        relu, up, dconv6, batch_norm2_1, relu, up, dconv7, batch_norm,
                                                        relu, up, dconv8, tanh)

class FrontaliseModelMasks_deeper(nn.Module):
        def __init__(self, input_nc, num_decoders=5, inner_nc=128, num_additional_ids=0, smaller=False, num_masks=0):
                super(FrontaliseModelMasks_deeper, self).__init__()
                self.encoder = self.generate_encoder_layers(output_size=inner_nc)
                
                self.decoder = self.generate_decoder_layers(inner_nc*2)
                self.mask = self.generate_decoder_layers(inner_nc*2, num_output_channels=1)


        def generate_encoder_layers(self, output_size=128, num_filters=16):
                conv1 = nn.Conv2d(3, num_filters, 3, 2, 1)
                conv2 = nn.Conv2d(num_filters, num_filters * 2, 3, 2, 1)
                conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 3, 2, 1)
                conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, 3, 2, 1)
                conv5 = nn.Conv2d(num_filters * 8, num_filters * 8, 3, 2, 1)
                conv6 = nn.Conv2d(num_filters * 8, num_filters * 8, 3, 2, 1)
                conv7 = nn.Conv2d(num_filters * 8, num_filters * 8, 3, 2, 1)
                conv8 = nn.Conv2d(num_filters * 8, num_filters * 8, 3, 2, 1)
                conv9 = nn.Conv2d(num_filters * 8, num_filters * 8, 3, 2, 1)
                conv10 = nn.Conv2d(num_filters * 8, output_size, 3, 2, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)

                leaky_relu = nn.LeakyReLU(0.2, True)
                return nn.Sequential(conv1, leaky_relu, conv2, batch_norm2_0, \
                                                          leaky_relu, conv3, batch_norm4_0, leaky_relu, \
                                                          conv4, batch_norm8_0, leaky_relu,
                                                          conv5, batch_norm8_1, leaky_relu,
                                                          conv6, batch_norm8_2, leaky_relu,
                                                          conv7, batch_norm8_3, leaky_relu,
                                                          conv8, batch_norm8_4, leaky_relu,
                                                          conv9, batch_norm8_5, leaky_relu,
                                                          conv10)



        def generate_decoder_layers(self, num_input_channels, num_output_channels=2, num_filters=16):
                up = nn.Upsample(scale_factor=2, mode='bilinear')

                dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
                dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 8, 3, 1, 1)
                dconv5 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
                dconv6 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
                dconv7 = nn.Conv2d(num_filters * 2 , num_filters, 3, 1, 1)
                dconv8 = nn.Conv2d(num_filters , num_output_channels, 3, 1, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm4_1 = nn.BatchNorm2d(num_filters * 4)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_7 = nn.BatchNorm2d(num_filters * 8)

                leaky_relu = nn.LeakyReLU(0.2)
                relu = nn.ReLU()
                tanh = nn.Tanh()

                return nn.Sequential(relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv1, batch_norm8_4, \
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv2, batch_norm8_5, relu,
                                                        nn.Upsample(scale_factor=2, mode='bilinear'), dconv3, batch_norm8_6, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv4,
                                                        batch_norm8_7, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv5, batch_norm4_1, 
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv6, batch_norm2_1, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv7, batch_norm,
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv8, tanh)
                


        def generate_decoder_small_layers(self, num_input_channels, num_output_channels=2, num_filters=16):
                up = nn.Upsample(scale_factor=2, mode='bilinear')

                dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
                dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
                dconv5 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
                dconv6 = nn.Conv2d(num_filters * 2 , num_output_channels, 3, 1, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm4_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_7 = nn.BatchNorm2d(num_filters * 4)

                leaky_relu = nn.LeakyReLU(0.2)
                relu = nn.ReLU()
                tanh = nn.Tanh()

                return nn.Sequential(relu, up, dconv1, batch_norm8_4, \
                                                        relu, up, dconv2, batch_norm8_5, relu,
                                                        up, dconv3, batch_norm8_6, relu, up, dconv4,
                                                        batch_norm8_7, relu, up, dconv5, batch_norm4_1, 
                                                        relu, up, dconv6, tanh)

class FrontaliseModelMasks_deeper_wider(nn.Module):
        def __init__(self, input_nc, num_decoders=5, inner_nc=128, num_additional_ids=0, smaller=False, num_masks=0):
                super(FrontaliseModelMasks_deeper_wider, self).__init__()
                self.encoder = self.generate_encoder_layers(output_size=inner_nc)
                
                self.decoder = self.generate_decoder_layers(inner_nc*2)
                self.mask = self.generate_decoder_layers(inner_nc*2, num_output_channels=1)


        def generate_encoder_layers(self, output_size=128, num_filters=32):
                conv1 = nn.Conv2d(3, num_filters, 3, 2, 1)
                conv2 = nn.Conv2d(num_filters, num_filters * 2, 3, 2, 1)
                conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 3, 2, 1)
                conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, 3, 2, 1)
                conv5 = nn.Conv2d(num_filters * 8, num_filters * 8, 3, 2, 1)
                conv6 = nn.Conv2d(num_filters * 8, num_filters * 8, 3, 2, 1)
                conv7 = nn.Conv2d(num_filters * 8, num_filters * 8, 3, 2, 1)
                conv8 = nn.Conv2d(num_filters * 8, num_filters * 8, 3, 2, 1)
                conv9 = nn.Conv2d(num_filters * 8, num_filters * 8, 3, 2, 1)
                conv10 = nn.Conv2d(num_filters * 8, output_size, 3, 2, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)

                leaky_relu = nn.LeakyReLU(0.2, True)
                return nn.Sequential(conv1, leaky_relu, conv2, batch_norm2_0, \
                                                          leaky_relu, conv3, batch_norm4_0, leaky_relu, \
                                                          conv4, batch_norm8_0, leaky_relu,
                                                          conv5, batch_norm8_1, leaky_relu,
                                                          conv6, batch_norm8_2, leaky_relu,
                                                          conv7, batch_norm8_3, leaky_relu,
                                                          conv8, batch_norm8_4, leaky_relu,
                                                          conv9, batch_norm8_5, leaky_relu,
                                                          conv10)



        def generate_decoder_layers(self, num_input_channels, num_output_channels=2, num_filters=16):
                up = nn.Upsample(scale_factor=2, mode='bilinear')

                dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
                dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 8, 3, 1, 1)
                dconv5 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
                dconv6 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
                dconv7 = nn.Conv2d(num_filters * 2 , num_filters, 3, 1, 1)
                dconv8 = nn.Conv2d(num_filters , num_output_channels, 3, 1, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm4_1 = nn.BatchNorm2d(num_filters * 4)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_7 = nn.BatchNorm2d(num_filters * 8)

                leaky_relu = nn.LeakyReLU(0.2)
                relu = nn.ReLU()
                tanh = nn.Tanh()

                return nn.Sequential(relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv1, batch_norm8_4, \
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv2, batch_norm8_5, relu,
                                                        nn.Upsample(scale_factor=2, mode='bilinear'), dconv3, batch_norm8_6, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv4,
                                                        batch_norm8_7, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv5, batch_norm4_1, 
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv6, batch_norm2_1, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv7, batch_norm,
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv8, tanh)
                


        def generate_decoder_small_layers(self, num_input_channels, num_output_channels=2, num_filters=16):
                up = nn.Upsample(scale_factor=2, mode='bilinear')

                dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
                dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
                dconv5 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
                dconv6 = nn.Conv2d(num_filters * 2 , num_output_channels, 3, 1, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm4_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_7 = nn.BatchNorm2d(num_filters * 4)

                leaky_relu = nn.LeakyReLU(0.2)
                relu = nn.ReLU()
                tanh = nn.Tanh()

                return nn.Sequential(relu, up, dconv1, batch_norm8_4, \
                                                        relu, up, dconv2, batch_norm8_5, relu,
                                                        up, dconv3, batch_norm8_6, relu, up, dconv4,
                                                        batch_norm8_7, relu, up, dconv5, batch_norm4_1, 
                                                        relu, up, dconv6, tanh)

class FrontaliseModelMasks_wider_hierarchy64(nn.Module):
        def __init__(self, input_nc, num_super_pixels=10, num_decoders=5, inner_nc=128, num_inner_samples=64*64, 
                                num_outer_samples=36,
                                num_additional_ids=0, smaller=False, num_masks=0, use_confidence=False,
                                tracking=False):
                super(FrontaliseModelMasks_wider_hierarchy64, self).__init__()
                
                if tracking:
                        #self.encoder = self.generate_encoder_layers(output_size=inner_nc, num_filters=num_additional_ids, tracking=tracking)[0:9]
                        #self.encoder[-1] = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                        self.encoder = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
                        self.encoder[3] = nn.MaxPool2d(kernel_size=1, stride=1, padding=0, ceil_mode=False)
                        self.encoder[0] = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
                        self.decoder = self.generate_decoder_layers(inner_nc, num_filters=num_additional_ids, num_output_channels=num_super_pixels)
                else:
                        self.encoder = self.generate_encoder_layers(output_size=inner_nc, num_filters=num_additional_ids, tracking=tracking, input_nc=input_nc)
                        self.decoder = self.generate_decoder_layers(inner_nc*2, num_filters=num_additional_ids, num_output_channels=num_super_pixels)
                
                self.sampler_decoder = nn.Sequential(
                        self.decoder[23:27],
                        self.decoder[27:])
               
                self.decoder = self.decoder[0:23] 
                if tracking:
                        self.decoder = nn.Sequential(
                                nn.Conv2d(1024+512, 256, kernel_size=5, stride=1, padding=2),
                                nn.LeakyReLU(0.2),
                                nn.BatchNorm2d(256),        
                                nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2))


                self.decoder_finetune = nn.Conv2d(64, num_inner_samples, 5, stride=1, padding=2)
                self.finetune_decoder = nn.Sequential(
                        nn.Conv2d(32,num_outer_samples,3,1,1),
                        nn.Conv2d(32,num_outer_samples,3,1,1))

                if use_confidence:
                        self.decoder_confidence = nn.Sequential(
                                nn.Conv2d(64, 1, 5, stride=1, padding=2),
                                nn.Softplus(beta=0.1))

        def generate_encoder_layers(self, input_nc=3, output_size=128, num_filters=64, tracking=False):
                conv1 = nn.Conv2d(input_nc, num_filters, 4, 2, 1)
                conv2 = nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1)
                conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1)
                conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1)
                conv5 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                if tracking:
                        conv6 = nn.Conv2d(512, num_filters * 8, 4, 2, 1)

                else:
                        conv6 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv7 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv8 = nn.Conv2d(num_filters * 8, output_size, 4, 2, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)

                leaky_relu = nn.LeakyReLU(0.2, True)
                return nn.Sequential(conv1, leaky_relu, conv2, batch_norm2_0, \
                                                          leaky_relu, conv3, batch_norm4_0, leaky_relu, \
                                                          conv4, batch_norm8_0, leaky_relu, conv5, 
                                                          batch_norm8_1, leaky_relu, conv6, batch_norm8_2, 
                                                          leaky_relu, conv7, batch_norm8_3, leaky_relu, conv8)

        def generate_decoder_layers(self, num_input_channels, num_output_channels=16, num_filters=32):
                up = nn.Upsample(scale_factor=2, mode='bilinear')

                dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
                dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 8, 3, 1, 1)
                dconv5 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
                dconv6 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
                dconv7 = nn.Conv2d(num_filters * 2 , num_filters, 3, 1, 1)
                dconv8 = nn.Conv2d(num_filters , num_output_channels, 3, 1, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm4_1 = nn.BatchNorm2d(num_filters * 4)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_7 = nn.BatchNorm2d(num_filters * 8)

                leaky_relu = nn.LeakyReLU(0.2)
                relu = nn.ReLU()

                return nn.Sequential(relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv1, batch_norm8_4, \
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv2, batch_norm8_5, relu,
                                                        nn.Upsample(scale_factor=2, mode='bilinear'), dconv3, batch_norm8_6, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv4,
                                                        batch_norm8_7, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv5, batch_norm4_1, 
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv6, batch_norm2_1, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv7, batch_norm,
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv8)
                


        def generate_decoder_small_layers(self, num_input_channels, num_output_channels=2, num_filters=16):
                up = nn.Upsample(scale_factor=2, mode='bilinear')

                dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
                dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
                dconv5 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
                dconv6 = nn.Conv2d(num_filters * 2 , num_output_channels, 3, 1, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm4_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_7 = nn.BatchNorm2d(num_filters * 4)

                leaky_relu = nn.LeakyReLU(0.2)
                relu = nn.ReLU()
                tanh = nn.Tanh()

                return nn.Sequential(relu, up, dconv1, batch_norm8_4, \
                                                        relu, up, dconv2, batch_norm8_5, relu,
                                                        up, dconv3, batch_norm8_6, relu, up, dconv4,
                                                        batch_norm8_7, relu, up, dconv5, batch_norm4_1, 
                                                        relu, up, dconv6, tanh)
class FrontaliseModelMasks_wider_hierarchy(nn.Module):
        def __init__(self, input_nc, num_super_pixels=10, num_decoders=5, inner_nc=128, num_inner_samples=32*32, 
                                num_outer_samples=36, 
                                num_additional_ids=0, smaller=False, num_masks=0, use_confidence=False, sep_dec_conf=False,
                                tracking=False):
                super(FrontaliseModelMasks_wider_hierarchy, self).__init__()
                
                if tracking:
                        #self.encoder = self.generate_encoder_layers(output_size=inner_nc, num_filters=num_additional_ids, tracking=tracking)[0:9]
                        #self.encoder[-1] = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                        self.encoder = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
                        self.encoder[3] = nn.MaxPool2d(kernel_size=1, stride=1, padding=0, ceil_mode=False)
                        self.encoder[0] = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
                        self.decoder = self.generate_decoder_layers(inner_nc, num_filters=num_additional_ids, num_output_channels=num_super_pixels)
                else:
                        self.encoder = self.generate_encoder_layers(output_size=inner_nc, num_filters=num_additional_ids, tracking=tracking, input_nc=input_nc)
                        self.decoder = self.generate_decoder_layers(inner_nc*2, num_filters=num_additional_ids, num_output_channels=num_super_pixels)
                
                self.k = (num_outer_samples) // 4
                print(self.k)
                self.sampler_decoder = nn.Sequential(
                        self.decoder[20:23],
                        self.decoder[23:27],
                        self.decoder[27:])
                #self.decoder = self.decoder[0:20]
                if tracking:
                        self.decoder = nn.Sequential(
                                nn.Conv2d(1024+512, 256, kernel_size=5, stride=1, padding=2),
                                nn.LeakyReLU(0.2),
                                nn.BatchNorm2d(256),        
                                nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2))


                self.decoder_finetune = nn.Conv2d(128, num_inner_samples, 5, stride=1, padding=2)
                self.finetune_decoder = nn.Sequential(
                        nn.Conv2d(64,num_outer_samples,3,1,1),
                        nn.Conv2d(32,num_outer_samples,3,1,1),
                        nn.Conv2d(32,num_outer_samples,3,1,1))
                self.sep_dec_conf = sep_dec_conf
                if use_confidence:
                        if self.sep_dec_conf:
                                self.confidence_decoder = self.generate_decoder_layers(inner_nc*2, num_filters=num_additional_ids, num_output_channels=1)
                                self.confidence_decoder = nn.Sequential(
                                        self.confidence_decoder[0:20], self.confidence_decoder[20:23], self.confidence_decoder[23:27], self.confidence_decoder[27:])

                                self.decoder_confidence = nn.Sequential(
                                        nn.Sequential(nn.Conv2d(128, 1, 5, stride=1, padding=2), nn.Softplus(beta=0.1)),
                                        nn.Sequential(nn.Conv2d(64,  1, 5, stride=1, padding=2), nn.Softplus(beta=0.1)),
                                        nn.Sequential(nn.Conv2d(32,  1, 5, stride=1, padding=2), nn.Softplus(beta=0.1)),
                                        nn.Sequential(nn.Conv2d(32,  1, 5, stride=1, padding=2), nn.Softplus(beta=0.1)))
                        else:
                                self.decoder_confidence = nn.Sequential(
                                nn.Conv2d(128, 1, 5, stride=1, padding=2),
                                nn.Softplus(beta=0.1))

        def generate_encoder_layers(self, input_nc=3, output_size=128, num_filters=64, tracking=False):
                conv1 = nn.Conv2d(input_nc, num_filters, 4, 2, 1)
                conv2 = nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1)
                conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1)
                conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1)
                conv5 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                if tracking:
                        conv6 = nn.Conv2d(512, num_filters * 8, 4, 2, 1)

                else:
                        conv6 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv7 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv8 = nn.Conv2d(num_filters * 8, output_size, 4, 2, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)

                leaky_relu = nn.LeakyReLU(0.2, True)
                return nn.Sequential(conv1, leaky_relu, conv2, batch_norm2_0, \
                                                          leaky_relu, conv3, batch_norm4_0, leaky_relu, \
                                                          conv4, batch_norm8_0, leaky_relu, conv5, 
                                                          batch_norm8_1, leaky_relu, conv6, batch_norm8_2, 
                                                          leaky_relu, conv7, batch_norm8_3, leaky_relu, conv8)

        def generate_decoder_layers(self, num_input_channels, num_output_channels=16, num_filters=32):
                up = nn.Upsample(scale_factor=2, mode='bilinear')

                dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
                dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 8, 3, 1, 1)
                dconv5 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
                dconv6 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
                dconv7 = nn.Conv2d(num_filters * 2 , num_filters, 3, 1, 1)
                dconv8 = nn.Conv2d(num_filters , num_output_channels, 3, 1, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm4_1 = nn.BatchNorm2d(num_filters * 4)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_7 = nn.BatchNorm2d(num_filters * 8)

                leaky_relu = nn.LeakyReLU(0.2)
                relu = nn.ReLU()

                return nn.Sequential(relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv1, batch_norm8_4, \
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv2, batch_norm8_5, relu,
                                                        nn.Upsample(scale_factor=2, mode='bilinear'), dconv3, batch_norm8_6, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv4,
                                                        batch_norm8_7, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv5, batch_norm4_1, 
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv6, batch_norm2_1, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv7, batch_norm,
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv8)
                


        def generate_decoder_small_layers(self, num_input_channels, num_output_channels=2, num_filters=16):
                up = nn.Upsample(scale_factor=2, mode='bilinear')

                dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
                dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
                dconv5 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
                dconv6 = nn.Conv2d(num_filters * 2 , num_output_channels, 3, 1, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm4_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_7 = nn.BatchNorm2d(num_filters * 4)

                leaky_relu = nn.LeakyReLU(0.2)
                relu = nn.ReLU()
                tanh = nn.Tanh()

                return nn.Sequential(relu, up, dconv1, batch_norm8_4, \
                                                        relu, up, dconv2, batch_norm8_5, relu,
                                                        up, dconv3, batch_norm8_6, relu, up, dconv4,
                                                        batch_norm8_7, relu, up, dconv5, batch_norm4_1, 
                                                        relu, up, dconv6, tanh)
class FrontaliseModelMasks_wider(nn.Module):
        def __init__(self, input_nc, num_super_pixels=10, num_decoders=5, inner_nc=128, num_additional_ids=0, smaller=False, num_masks=0, ind_imgs=False):
                super(FrontaliseModelMasks_wider, self).__init__()
                self.encoder = self.generate_encoder_layers(output_size=inner_nc, num_filters=num_additional_ids)
                
                if ind_imgs:
                        self.decoder = self.generate_decoder_layers(inner_nc, num_filters=num_additional_ids, num_output_channels=num_super_pixels)
                else:
                        self.decoder = self.generate_decoder_layers(inner_nc*2, num_filters=num_additional_ids, num_output_channels=num_super_pixels)
                        self.decoder_finetune = nn.Conv2d(128, 32*32, 5, stride=1, padding=2)

                self.affine_decoder = nn.Sequential(nn.Linear(inner_nc*2,128), nn.BatchNorm1d(128), nn.ReLU(),
                                                    nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
                                                    nn.Linear(64, 6 * num_super_pixels))


        def generate_encoder_layers(self, output_size=128, num_filters=64):
                conv1 = nn.Conv2d(3, num_filters, 4, 2, 1)
                conv2 = nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1)
                conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1)
                conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1)
                conv5 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv6 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv7 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv8 = nn.Conv2d(num_filters * 8, output_size, 4, 2, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)

                leaky_relu = nn.LeakyReLU(0.2, True)
                return nn.Sequential(conv1, leaky_relu, conv2, batch_norm2_0, \
                                                          leaky_relu, conv3, batch_norm4_0, leaky_relu, \
                                                          conv4, batch_norm8_0, leaky_relu, conv5, 
                                                          batch_norm8_1, leaky_relu, conv6, batch_norm8_2, 
                                                          leaky_relu, conv7, batch_norm8_3, leaky_relu, conv8)

        def generate_decoder_layers(self, num_input_channels, num_output_channels=2, num_filters=32):
                up = nn.Upsample(scale_factor=2, mode='bilinear')

                dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
                dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 8, 3, 1, 1)
                dconv5 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
                dconv6 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
                dconv7 = nn.Conv2d(num_filters * 2 , num_filters, 3, 1, 1)
                dconv8 = nn.Conv2d(num_filters , num_output_channels, 3, 1, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm4_1 = nn.BatchNorm2d(num_filters * 4)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_7 = nn.BatchNorm2d(num_filters * 8)

                leaky_relu = nn.LeakyReLU(0.2)
                relu = nn.ReLU()

                return nn.Sequential(relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv1, batch_norm8_4, \
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv2, batch_norm8_5, relu,
                                                        nn.Upsample(scale_factor=2, mode='bilinear'), dconv3, batch_norm8_6, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv4,
                                                        batch_norm8_7, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv5, batch_norm4_1, 
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv6, batch_norm2_1, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv7, batch_norm,
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv8)
                


        def generate_decoder_small_layers(self, num_input_channels, num_output_channels=2, num_filters=16):
                up = nn.Upsample(scale_factor=2, mode='bilinear')

                dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
                dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
                dconv5 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
                dconv6 = nn.Conv2d(num_filters * 2 , num_output_channels, 3, 1, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm4_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_7 = nn.BatchNorm2d(num_filters * 4)

                leaky_relu = nn.LeakyReLU(0.2)
                relu = nn.ReLU()
                tanh = nn.Tanh()

                return nn.Sequential(relu, up, dconv1, batch_norm8_4, \
                                                        relu, up, dconv2, batch_norm8_5, relu,
                                                        up, dconv3, batch_norm8_6, relu, up, dconv4,
                                                        batch_norm8_7, relu, up, dconv5, batch_norm4_1, 
                                                        relu, up, dconv6, tanh)


class FrontaliseModelLighting(nn.Module):
        def __init__(self, input_nc, num_decoders=5, inner_nc=128, num_additional_ids=0, smaller=False, num_masks=0):
                super(FrontaliseModelLighting, self).__init__()
                print(num_additional_ids, inner_nc)
                self.encoder = self.generate_encoder_layers(output_size=inner_nc, num_filters=num_additional_ids)
                
                self.decoder = self.generate_decoder_layers(inner_nc*2, num_filters=num_additional_ids)
                self.mask = self.generate_decoder_layers(inner_nc*2, num_output_channels=1, num_filters=num_additional_ids)
                self.lighting = self.generate_decoder_layers(4, num_output_channels=1, num_filters=num_additional_ids)


        def generate_encoder_layers(self, output_size=128, num_filters=64):
                conv1 = nn.Conv2d(3, num_filters, 4, 2, 1)
                conv2 = nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1)
                conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1)
                conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1)
                conv5 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv6 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv7 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
                conv8 = nn.Conv2d(num_filters * 8, output_size, 4, 2, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)

                leaky_relu = nn.LeakyReLU(0.2, True)
                return nn.Sequential(conv1, leaky_relu, conv2, batch_norm2_0, \
                                                          leaky_relu, conv3, batch_norm4_0, leaky_relu, \
                                                          conv4, batch_norm8_0, leaky_relu, conv5, 
                                                          batch_norm8_1, leaky_relu, conv6, batch_norm8_2, 
                                                          leaky_relu, conv7, batch_norm8_3, leaky_relu, conv8)

        def generate_decoder_layers(self, num_input_channels, num_output_channels=2, num_filters=32):
                up = nn.Upsample(scale_factor=2, mode='bilinear')

                dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
                dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 8, 3, 1, 1)
                dconv5 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
                dconv6 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
                dconv7 = nn.Conv2d(num_filters * 2 , num_filters, 3, 1, 1)
                dconv8 = nn.Conv2d(num_filters , num_output_channels, 3, 1, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm4_1 = nn.BatchNorm2d(num_filters * 4)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_7 = nn.BatchNorm2d(num_filters * 8)

                leaky_relu = nn.LeakyReLU(0.2)
                relu = nn.ReLU()
                tanh = nn.Tanh()

                return nn.Sequential(relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv1, batch_norm8_4, \
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv2, batch_norm8_5, relu,
                                                        nn.Upsample(scale_factor=2, mode='bilinear'), dconv3, batch_norm8_6, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv4,
                                                        batch_norm8_7, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv5, batch_norm4_1, 
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv6, batch_norm2_1, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv7, batch_norm,
                                                        relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv8, tanh)
                


        def generate_decoder_small_layers(self, num_input_channels, num_output_channels=2, num_filters=16):
                up = nn.Upsample(scale_factor=2, mode='bilinear')

                dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
                dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
                dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
                dconv5 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
                dconv6 = nn.Conv2d(num_filters * 2 , num_output_channels, 3, 1, 1)

                batch_norm = nn.BatchNorm2d(num_filters)
                batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
                batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
                batch_norm4_1 = nn.BatchNorm2d(num_filters * 2)
                batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
                batch_norm8_7 = nn.BatchNorm2d(num_filters * 4)

                leaky_relu = nn.LeakyReLU(0.2)
                relu = nn.ReLU()
                tanh = nn.Tanh()

                return nn.Sequential(relu, up, dconv1, batch_norm8_4, \
                                                        relu, up, dconv2, batch_norm8_5, relu,
                                                        up, dconv3, batch_norm8_6, relu, up, dconv4,
                                                        batch_norm8_7, relu, up, dconv5, batch_norm4_1, 
                                                        relu, up, dconv6, tanh)
