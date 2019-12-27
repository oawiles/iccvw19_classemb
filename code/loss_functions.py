import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

class FeatureLoss(nn.Module):
        def __init__(self, num_classes=10):
                super(FeatureLoss, self).__init__()
                self.feature_model = torchvision.models.resnet18(pretrained=True)
                self.feature_model = nn.Sequential(*list(self.feature_model.children())[:-6])
                self.num_classes = num_classes

        def forward(self, img1, class_im1):
                b, _, w, h = class_im1.size()

                # Get features and downscale class probabilities
                img1 = self.feature_model(img1)
                _, c, _, _ = img1.size()
                img1 = nn.Upsample(size=(w,h))(img1)

                loss = torch.zeros((1,)).cuda()
                for i in range(0, class_im1.size(1)):
                        feats = img1.view(b, c, -1)           
                        weights = class_im1[:,i:i+1,:,:].view(b,1,-1)
                        mean_feat = (feats * weights).sum(dim=2, keepdim=True) / weights.sum(dim=2, keepdim=True)

                        err = (feats - mean_feat).pow(2).sum(dim=1, keepdim=True) * weights
                        loss += err.mean(dim=2).mean()

                return loss / float(class_im1.size(1))  

def multiview_maskfaceloss(input_image, model, front_images):
        assert(len(front_images) == 1)
        b, c, w, h = input_image.size()

        # Predict the classes
        inp_emb = model.encoder(input_image)
        out_emb = model.encoder(front_images[0])

        emb = torch.cat((inp_emb, out_emb), 1)
        classes = model.decoder(emb)

        # Get probability per class
        classes = nn.Softmax2d()(classes)

        affine_decoders = model.affine_decoder(emb.squeeze()).view(b, -1, 2, 3).contiguous()

        # Add offset to affine trans
        offset = torch.eye(3)[0:2,:].unsqueeze(0).unsqueeze(0).cuda()
        affine_decoders = affine_decoders + offset
        

        # And then do transformations + probabilities
        trans_image = front_images[0].unsqueeze(1).repeat(1, affine_decoders.size(1), 1, 1, 1)
        trans_image = trans_image.view(-1,c,w,h) 
        affine_decoders = affine_decoders.view(-1, 2, 3).contiguous()
       
        affine_grid = F.affine_grid(affine_decoders, size=(affine_decoders.size(0), 3, w, h)) 
        gen_image = F.grid_sample(trans_image, affine_grid)

        # Now compare
        loss_image = gen_image.view(b, -1, c, w, h).contiguous() - input_image.unsqueeze(1)
        loss_image = loss_image * classes.unsqueeze(2)

        loss = loss_image.abs().sum(dim=2).view(b, -1, w * h).contiguous().mean(dim=2).sum(dim=1).mean()

        result_image = gen_image.view(b, -1, c, w, h) * classes.unsqueeze(2)
        result_image = result_image.sum(dim=1).contiguous().detach()
        return loss, result_image, classes.detach()

def multiview_maskfaceloss_probs(input_image, model, front_images, prev_probs, size=(256,256)):
        assert(len(front_images) == 1)
        b, c, w, h = input_image.size()

        # Predict the classes
        inp_emb = model.encoder(input_image)
        out_emb = model.encoder(front_images[0])

        emb = torch.cat((inp_emb, out_emb), 1)
        classes = model.decoder(emb)

        # Get probability per class
        classes = nn.Softmax2d()(classes)

        affine_decoders = model.affine_decoder(emb.squeeze()).view(b, -1, 2, 3).contiguous()

        # Add offset to affine trans
        offset = torch.eye(3)[0:2,:].unsqueeze(0).unsqueeze(0).cuda()
        affine_decoders = affine_decoders + offset
        

        # And then do transformations + probabilities
        trans_image = front_images[0].unsqueeze(1).repeat(1, affine_decoders.size(1), 1, 1, 1)
        trans_image = trans_image.view(-1,c,w,h) 
        affine_decoders = affine_decoders.view(-1, 2, 3).contiguous()
       
        affine_grid = F.affine_grid(affine_decoders, size=(affine_decoders.size(0), 3, w, h)) 
        gen_image = F.grid_sample(trans_image, affine_grid)

        # Now compare
        loss_image = gen_image.view(b, -1, c, w, h).contiguous() - input_image.unsqueeze(1)
        loss_image = loss_image * classes.unsqueeze(2)

        loss = loss_image.abs().sum(dim=2).view(b, -1, w * h).contiguous().mean(dim=2).sum(dim=1).mean()

        result_image = gen_image.view(b, -1, c, w, h) * classes.unsqueeze(2)
        result_image = result_image.sum(dim=1).contiguous().detach()
        
        # Now transform the probabilities:
        probabilities = prev_probs.view(-1, 1, w, h)
        affine_grid = F.affine_grid(affine_decoders, size=(affine_decoders.size(0), 1, w, h))
        gen_probs = F.grid_sample(probabilities, affine_grid)
        gen_probs = gen_probs.view(b, -1, w, h)

        # And compare
        loss_probs = (gen_probs - classes).abs().view(b, -1).mean(dim=1).mean()
        return loss, loss_probs, result_image, classes.detach()

def multiview_hierarchy_old(input_image, model, front_images, size=(256,256)):
        assert(len(front_images) == 1)
        b, c, w, h = input_image.size()

        # Predict the classes
        inp_emb = model.encoder(input_image)
        out_emb = model.encoder(front_images[0])

        emb = torch.cat((inp_emb, out_emb), 1)
        orig_probs = model.decoder[0:20](emb)
        probs_fine = model.decoder_finetune(orig_probs)

        # Calculate low-level probability
        probs = nn.Softmax2d()(probs_fine)
        b, c, w, h = probs.size()
        probs = probs.unsqueeze(1)
        fr_img = nn.Upsample(size=(w,h))(front_images[0])
        in_img = nn.Upsample(size=(w,h))(input_image)

        genimg_lowlevel = probs * fr_img.view(b,3,-1).unsqueeze(3).unsqueeze(4)
        genimg_lowlevel = genimg_lowlevel.sum(dim=2).view(b,3,w,h)
        loss_lowlevel = (genimg_lowlevel - in_img).abs().sum(dim=1).view(b,-1).mean(dim=1).mean()
       
        # Get higher level probability 
        topk = probs.squeeze().topk(dim=1, k=9)
       
        upper_probs = model.sampler_decoder(orig_probs)
        locs = topk[1]
        with torch.no_grad():
                w = (locs % 32).float() * 2 
                h = (locs // 32).float() * 2
                w = w.unsqueeze(2).repeat(1,1,4,1,1)
                w[:,:,2:,:,:] = w[:,:,2:,:,:] + 1
                h = h.unsqueeze(2).repeat(1,1,4,1,1)
                h[:,:,[1,3],:,:] = h[:,:,[1,3],:,:] + 1
        w = w / 32. - 1
        h = h / 32. - 1

        w = nn.Upsample(size=(64,64), mode='nearest')(w.view(-1, 36, 32, 32))
        h = nn.Upsample(size=(64,64), mode='nearest')(h.view(-1, 36, 32, 32))
        upper_probs = nn.Softmax2d()(upper_probs)

        w = w.view(-1,64,64,1)
        h = h.view(-1,64,64,1)
        sampler = torch.cat((w, h), 3)
        with torch.no_grad():
                imgs = front_images[0].unsqueeze(1).repeat(1,36,1,1,1).view(-1,3,256,256)
                sampled_imgs = F.grid_sample(imgs, sampler)
                sampled_imgs = sampled_imgs.view(-1,36,3,64,64)

        gen_image = (sampled_imgs * upper_probs.unsqueeze(2)).sum(dim=1)
        in_img = nn.Upsample(size=size)(input_image)
        gen_loss = (gen_image - in_img).abs().sum(dim=1).view(b,-1).mean(dim=1).mean()

        loss_identity = torch.zeros((1,)).cuda()
        return gen_loss, loss_lowlevel, loss_identity, probs, gen_image 

def l1_withconf(genimg_lowlevel, in_img, conf):
        b, c, w, h = in_img.size()
        conf = conf.clamp(min=1e-2)
        err = 1.4142 * (genimg_lowlevel - in_img).abs().sum(dim=1, keepdim=True)

        loss = (err / conf) - (1.4142 / (2. * conf)).log()

        loss = loss.view(b, -1).mean(dim=1).mean()

        #err = (genimg_lowlevel - in_img).abs().sum(dim=1).view(b, -1).mean(dim=1).mean()
        #return err
        if not(loss == loss):
                import pdb
                pdb.set_trace()
                print(conf.min(), conf.max(), err.min(), err.max())
                print(1+'1')

        return loss        

def multiview_hierarchy_sampler_confidence(input_image, model, front_images, size=(256,256)):
        b, c, w, h = input_image.size()

        # Predict the classes
        inp_emb = model.encoder(input_image)
        out_emb = model.encoder(front_images[0])
        emb = torch.cat((inp_emb, out_emb), 1)
        orig_probs = model.decoder[0:20](emb)
        probs_fine = nn.Tanh()(model.decoder_finetune(orig_probs) * 0.1)
        conf = model.decoder_confidence(orig_probs)

        # Calculate low-level coordinates
        b, c, w, h = probs_fine.size()
        coors = (probs_fine).permute(0,2,3,1) + F.affine_grid(torch.eye(3)[0:2,:].unsqueeze(0).repeat(b, 1, 1).cuda(), size=torch.Size((b,1,w,h)))
        in_img = nn.Upsample(size=(w,h))(input_image)

        coorslowlevel = coors
        genimg_lowlevel = F.grid_sample(front_images[0], coors)
        loss_lowlevel = l1_withconf(genimg_lowlevel, in_img, conf)
        # Get higher level probability

        sampler = None; probs = None
        gen_loss = torch.zeros((1,)).cuda()      
        gen_image = genimg_lowlevel 
        index = 0
        while w < size[0]:
                orig_probs = model.sampler_decoder[index](orig_probs)
                ncoors = nn.Tanh()(model.finetune_decoder[index](orig_probs) * 0.1)

                w = w * 2; h = h * 2
                coors = nn.Upsample(size=(w,h))(coors.permute(0,3,1,2)).permute(0,2,3,1)
                coors = coors + ncoors.permute(0,2,3,1)
                gen_image = F.grid_sample(front_images[0], coors) 
                in_img = nn.Upsample(size=(w,h))(input_image)
                gen_loss += (gen_image - in_img).abs().sum(dim=1).view(b,-1).mean(dim=1).mean()

                index += 1
        
        loss_identity = torch.zeros((1,)).cuda()
        return gen_loss, loss_lowlevel, loss_identity, coorslowlevel, gen_image, (sampler, probs, conf) 

def multiview_hierarchy_confidence_multiview(input_image, model, front_images, size=(256,256), tracking=False):
        b, img_c, w, h = input_image.size()

        # Predict the classes

        orig_probs_all = []
        confs_all = []
        orig_confs_all = []
        probs_all = []
        genimgs_all = []
        for i in range(0, len(front_images)):
                inp_emb = model.encoder(input_image)
                out_emb = model.encoder(front_images[i])

                emb = torch.cat((inp_emb, out_emb), 1)
                orig_probs = model.decoder[0:20](emb)
                orig_probs_all += [orig_probs]

                probs_fine = model.decoder_finetune(orig_probs)
                if model.sep_dec_conf:
                        origconf = model.confidence_decoder[0](emb)
                        conf = model.decoder_confidence[0](origconf)
                
                else:
                        conf = model.decoder_confidence(orig_probs)
                confs_all += [conf]
                orig_confs_all += [origconf]
                conf_lowres = conf

                # Calculate low-level probability
                probs = nn.Softmax2d()(probs_fine)
                b, c, w, h = probs.size()
                #print(probs.size())
                probs = probs.unsqueeze(1)
                probs_all += [probs]
                fr_img = nn.Upsample(size=(w,h))(front_images[0])
                in_img = nn.Upsample(size=(w,h))(input_image)

                probs_lowlevel = probs
                genimg_lowlevel = probs * fr_img.view(b,img_c,-1).unsqueeze(3).unsqueeze(4)
                genimg_lowlevel = genimg_lowlevel.sum(dim=2).view(b,img_c,w,h)
                genimgs_all += [genimg_lowlevel]

        confs_all = torch.cat([c.unsqueeze(0) for c in confs_all], 0)
        conf, conf_ind = torch.min(confs_all, 0, keepdim=True)
        genimgs_all = torch.cat([g.unsqueeze(0) for g in genimgs_all], 0)
        genimg_lowlevel = genimgs_all.gather(index=conf_ind.repeat(1,1,3,1,1), dim=0).squeeze()

        loss_lowlevel = l1_withconf(genimg_lowlevel, in_img, conf)
        
        # Get higher level probability
        topk = [p.squeeze().topk(dim=1, k=model.k) for p in probs_all]
        locs = [t[1] for t in topk]
        wind = [(l % w).float() for l in locs] 
        hind = [(l // h).float() for l in locs] 

        sampler = None; probs = None
        gen_loss = torch.zeros((1,)).cuda()      
        gen_image = genimg_lowlevel 
        index = 0
        while w < size[0]:
                orig_probs_n = []
                gen_imgs = []
                probs_all = []
                w = w * 2; h = h * 2
                for i in range(0, len(front_images)):
                        orig_probs = model.sampler_decoder[index](orig_probs_all[i])
                        orig_probs_n += [orig_probs]
                        probs = model.finetune_decoder[index](orig_probs)
                        wind[i] = wind[i] * 2
                        hind[i] = hind[i] * 2
                        with torch.no_grad():
                                wind[i] = wind[i].unsqueeze(2).repeat(1,1,4,1,1)
                                wind[i][:,:,2:,:,:] = wind[i][:,:,2:,:,:] + 1
                                hind[i] = hind[i].unsqueeze(2).repeat(1,1,4,1,1)
                                hind[i][:,:,[1,3],:,:] = hind[i][:,:,[1,3],:,:] + 1

                        topk_p = topk[i][0].unsqueeze(2).repeat(1,1,4,1,1)

                        wind[i] = nn.Upsample(size=(w, h), mode='nearest')(wind[i].view(-1, model.k*4, w // 2, h // 2))
                        hind[i] = nn.Upsample(size=(w, h), mode='nearest')(hind[i].view(-1, model.k*4, w // 2, h // 2))
                        topk_p = nn.Upsample(size=(w,h), mode='nearest')(topk_p.view(-1, model.k*4, w // 2, h // 2))
                        probs = nn.Softmax2d()(probs * topk_p)
                        probs_all += [probs]
                        windt = wind[i].view(-1,w,h,1)
                        hindt = hind[i].view(-1,w,h,1)
                        windt = windt / float(w // 2.) - 1
                        hindt = hindt / float(h // 2.) - 1
                        sampler = torch.cat((windt, hindt), 3)
                        with torch.no_grad():
                                imgs = front_images[i].unsqueeze(1).repeat(1,model.k*4,1,1,1).view(-1,img_c,256,256)
                                sampled_imgs = F.grid_sample(imgs, sampler)
                                        
                                sampled_imgs = sampled_imgs.view(-1,model.k*4,img_c,w,h)
                        
                        gen_image = (sampled_imgs * probs_all[i].unsqueeze(2)).sum(dim=1)
                        gen_imgs += [gen_image]
                in_img = nn.Upsample(size=(w,h))(input_image)
                orig_probs_all = orig_probs_n
                confs_all = []
                for i in range(0, len(front_images)):
                        if model.sep_dec_conf:
                                orig_confs_all[i] = model.confidence_decoder[index+1](orig_confs_all[i])
                                confs_all += [model.decoder_confidence[index+1](orig_confs_all[i])]

                confs_all = torch.cat([c.unsqueeze(0) for c in confs_all], 0)
                conf, conf_ind = torch.min(confs_all, 0, keepdim=True)
                genimgs_all = torch.cat([g.unsqueeze(0) for g in gen_imgs], 0)
                gen_image = genimgs_all.gather(index=conf_ind.repeat(1,1,3,1,1), dim=0).squeeze()

                gen_loss += l1_withconf(gen_image, in_img, conf)

                # Get the topk indices
                topk = [p.topk(dim=1, k=model.k) for p in probs_all]
                wind = [wind[w].gather(index=topk[w][1], dim=1) for w in range(0, len(wind))]
                hind = [hind[h].gather(index=topk[h][1], dim=1) for h in range(0, len(hind))]
                index += 1
        
        loss_identity = torch.zeros((1,)).cuda()
        return gen_loss, loss_lowlevel, loss_identity, probs_lowlevel, gen_image, (sampler, probs, conf_lowres) 

def multiview_hierarchy_confidence_getresults(input_image, model, front_images, size=(256,256), tracking=False):
        assert(len(front_images) == 1)
        b, img_c, w, h = input_image.size()

        # Predict the classes
        inp_emb = model.encoder(input_image)
        out_emb = model.encoder(front_images[0])

        emb = torch.cat((inp_emb, out_emb), 1)
        orig_probs = model.decoder[0:20](emb)

        all_confs = []; all_genimgs = []

        probs_fine = model.decoder_finetune(orig_probs)
        origconf = model.confidence_decoder[0](emb)
        conf = model.decoder_confidence[0](origconf)
        conf_lowres = conf
        all_confs += [conf]

        # Calculate low-level probability
        probs = nn.Softmax2d()(probs_fine)
        b, c, w, h = probs.size()
        probs = probs.unsqueeze(1)
        fr_img = nn.Upsample(size=(w,h))(front_images[0])
        in_img = nn.Upsample(size=(w,h))(input_image)

        probs_lowlevel = probs
        genimg_lowlevel = probs * fr_img.view(b,img_c,-1).unsqueeze(3).unsqueeze(4)
        genimg_lowlevel = genimg_lowlevel.sum(dim=2).view(b,img_c,w,h)
        all_genimgs += [genimg_lowlevel]

        loss_lowlevel = l1_withconf(genimg_lowlevel, in_img, conf)
        # Get higher level probability
        topk = probs.squeeze().topk(dim=1, k=model.k)
        locs = topk[1]
        wind = (locs % w).float() 
        hind = (locs // h).float() 

        sampler = None; probs = None
        gen_loss = torch.zeros((1,)).cuda()      
        gen_image = genimg_lowlevel 
        index = 0
        while w < size[0]:
                orig_probs = model.sampler_decoder[index](orig_probs)
                probs = model.finetune_decoder[index](orig_probs)
                wind = wind * 2
                hind = hind * 2
                with torch.no_grad():
                        wind = wind.unsqueeze(2).repeat(1,1,4,1,1)
                        wind[:,:,2:,:,:] = wind[:,:,2:,:,:] + 1
                        hind = hind.unsqueeze(2).repeat(1,1,4,1,1)
                        hind[:,:,[1,3],:,:] = hind[:,:,[1,3],:,:] + 1

                topk_p = topk[0].unsqueeze(2).repeat(1,1,4,1,1)

                w = w * 2; h = h * 2
                wind = nn.Upsample(size=(w, h), mode='nearest')(wind.view(-1, model.k*4, w // 2, h // 2))
                hind = nn.Upsample(size=(w, h), mode='nearest')(hind.view(-1, model.k*4, w // 2, h // 2))
                topk_p = nn.Upsample(size=(w,h), mode='nearest')(topk_p.view(-1, model.k*4, w // 2, h // 2))
                probs = nn.Softmax2d()(probs * topk_p)

                windt = wind.view(-1,w,h,1)
                hindt = hind.view(-1,w,h,1)
                windt = windt / float(w // 2.) - 1
                hindt = hindt / float(h // 2.) - 1
                sampler = torch.cat((windt, hindt), 3)
                with torch.no_grad():
                        imgs = front_images[0].unsqueeze(1).repeat(1,model.k*4,1,1,1).view(-1,img_c,256,256)
                        sampled_imgs = F.grid_sample(imgs, sampler)
                                
                        sampled_imgs = sampled_imgs.view(-1,model.k*4,img_c,w,h)
                
                gen_image = (sampled_imgs * probs.unsqueeze(2)).sum(dim=1)
                in_img = nn.Upsample(size=(w,h))(input_image)

                if model.sep_dec_conf:
                        origconf = model.confidence_decoder[index+1](origconf)
                        conf = model.decoder_confidence[index+1](origconf)
                        all_confs += [conf]; all_genimgs += [gen_image]
                        gen_loss += l1_withconf(gen_image, in_img, conf)

                else:
                        gen_loss += l1_withconf(gen_image, in_img, nn.Upsample(size=(w,h))(conf))

                # Get the topk indices
                topk = probs.topk(dim=1, k=model.k)
                wind = wind.gather(index=topk[1], dim=1)
                hind = hind.gather(index=topk[1], dim=1)
                index += 1
        
        loss_identity = torch.zeros((1,)).cuda()
        return gen_loss, loss_lowlevel, loss_identity, probs_lowlevel, gen_image, (sampler, probs, all_confs, all_genimgs) 

def multiview_hierarchy_confidence(input_images, model, front_imagess, size=(256,256), tracking=False):
        # Predict the classes
        input_image = input_images[0]
        input_imagejit = input_images[1]
        front_images = front_imagess[0]
        front_imagesjit = front_imagess[1]

        b, img_c, w, h = input_image.size()
        assert(len(front_images) == 1)
        inp_emb = model.encoder(input_imagejit)
        out_emb = model.encoder(front_imagesjit[0])

        emb = torch.cat((inp_emb, out_emb), 1)
        orig_probs = model.decoder[0:20](emb)

        probs_fine = model.decoder_finetune(orig_probs)
        if model.sep_dec_conf:
                origconf = model.confidence_decoder[0](emb)
                conf = model.decoder_confidence[0](origconf)
        
        else:
                conf = model.decoder_confidence(orig_probs)
        conf_lowres = conf

        # Calculate low-level probability
        probs = nn.Softmax2d()(probs_fine)
        b, c, w, h = probs.size()
        probs = probs.unsqueeze(1)
        fr_img = nn.Upsample(size=(w,h))(front_images[0])
        in_img = nn.Upsample(size=(w,h))(input_image)

        probs_lowlevel = probs
        genimg_lowlevel = probs * fr_img.view(b,img_c,-1).unsqueeze(3).unsqueeze(4)
        genimg_lowlevel = genimg_lowlevel.sum(dim=2).view(b,img_c,w,h)
        loss_lowlevel = l1_withconf(genimg_lowlevel, in_img, conf)
        # Get higher level probability
        topk = probs.squeeze().topk(dim=1, k=model.k)
        locs = topk[1]
        wind = (locs % w).float() 
        hind = (locs // h).float() 

        sampler = None; probs = None
        gen_loss = torch.zeros((1,)).cuda()      
        gen_image = genimg_lowlevel 
        index = 0
        while w < size[0]:
                #print(orig_probs.size(), model.sampler_decoder[index])
                orig_probs = model.sampler_decoder[index](orig_probs)
                probs = model.finetune_decoder[index](orig_probs)
                wind = wind * 2
                hind = hind * 2
                with torch.no_grad():
                        wind = wind.unsqueeze(2).repeat(1,1,4,1,1)
                        wind[:,:,2:,:,:] = wind[:,:,2:,:,:] + 1
                        hind = hind.unsqueeze(2).repeat(1,1,4,1,1)
                        hind[:,:,[1,3],:,:] = hind[:,:,[1,3],:,:] + 1

                topk_p = topk[0].unsqueeze(2).repeat(1,1,4,1,1)

                w = w * 2; h = h * 2
                wind = nn.Upsample(size=(w, h), mode='nearest')(wind.view(-1, model.k*4, w // 2, h // 2))
                hind = nn.Upsample(size=(w, h), mode='nearest')(hind.view(-1, model.k*4, w // 2, h // 2))
                topk_p = nn.Upsample(size=(w,h), mode='nearest')(topk_p.view(-1, model.k*4, w // 2, h // 2))
                probs = nn.Softmax2d()(probs * topk_p)

                windt = wind.view(-1,w,h,1)
                hindt = hind.view(-1,w,h,1)
                windt = windt / float(w // 2.) - 1
                hindt = hindt / float(h // 2.) - 1
                sampler = torch.cat((windt, hindt), 3)
                with torch.no_grad():
                        imgs = front_images[0].unsqueeze(1).repeat(1,model.k*4,1,1,1).view(-1,img_c,256,256)
                        sampled_imgs = F.grid_sample(imgs, sampler)
                                
                        sampled_imgs = sampled_imgs.view(-1,model.k*4,img_c,w,h)
                
                gen_image = (sampled_imgs * probs.unsqueeze(2)).sum(dim=1)
                in_img = nn.Upsample(size=(w,h))(input_image)

                if model.sep_dec_conf:
                        origconf = model.confidence_decoder[index+1](origconf)
                        conf = model.decoder_confidence[index+1](origconf)
                        gen_loss += l1_withconf(gen_image, in_img, conf)

                else:
                        gen_loss += l1_withconf(gen_image, in_img, nn.Upsample(size=(w,h))(conf))

                # Get the topk indices
                topk = probs.topk(dim=1, k=model.k)
                wind = wind.gather(index=topk[1], dim=1)
                hind = hind.gather(index=topk[1], dim=1)
                index += 1
        
        loss_identity = torch.zeros((1,)).cuda()
        return gen_loss, loss_lowlevel, loss_identity, probs_lowlevel, gen_image, (sampler, probs, conf_lowres) 

def multiview_hierarchy(input_image, model, front_images, size=(256,256), tracking=False):
        assert(len(front_images) == 1)
        b, img_c, w, h = input_image.size()

        # Predict the classes
        if not(tracking):
                inp_emb = model.encoder(input_image)
                out_emb = model.encoder(front_images[0])

                emb = torch.cat((inp_emb, out_emb), 1)
                orig_probs = model.decoder[0:20](emb)
        else:
                originp_emb = model.encoder(input_image)
                inp_emb = originp_emb.permute(0,2,3,1).contiguous()
                out_emb = model.encoder(front_images[0]).permute(0,2,3,1).contiguous()
                b, w, h, c = inp_emb.size()
                # Compare features
                corr = out_emb.view(b,-1,c).bmm(inp_emb.view(b,-1,c).permute(0,2,1))
                corr = corr.view(b,w*h,w,h)
                orig_probs = model.decoder(torch.cat((corr, originp_emb), 1))

        probs_fine = model.decoder_finetune(orig_probs)

        # Calculate low-level probability
        probs = nn.Softmax2d()(probs_fine)
        b, c, w, h = probs.size()
        probs = probs.unsqueeze(1)
        fr_img = nn.Upsample(size=(w,h))(front_images[0])
        in_img = nn.Upsample(size=(w,h))(input_image)

        probs_lowlevel = probs
        genimg_lowlevel = probs * fr_img.view(b,img_c,-1).unsqueeze(3).unsqueeze(4)
        genimg_lowlevel = genimg_lowlevel.sum(dim=2).view(b,img_c,w,h)
        loss_lowlevel = (genimg_lowlevel - in_img).abs().sum(dim=1).view(b,-1).mean(dim=1).mean()
        # Get higher level probability
        topk = probs.squeeze().topk(dim=1, k=9)
        locs = topk[1]
        wind = (locs % w).float() 
        hind = (locs // h).float() 

        sampler = None; probs = None
        gen_loss = torch.zeros((1,)).cuda()      
        gen_image = genimg_lowlevel 
        index = 0
        while w < size[0]:
                orig_probs = model.sampler_decoder[index](orig_probs)
                probs = model.finetune_decoder[index](orig_probs)
                wind = wind * 2
                hind = hind * 2
                with torch.no_grad():
                        wind = wind.unsqueeze(2).repeat(1,1,4,1,1)
                        wind[:,:,2:,:,:] = wind[:,:,2:,:,:] + 1
                        hind = hind.unsqueeze(2).repeat(1,1,4,1,1)
                        hind[:,:,[1,3],:,:] = hind[:,:,[1,3],:,:] + 1

                topk_p = topk[0].unsqueeze(2).repeat(1,1,4,1,1)

                w = w * 2; h = h * 2
                wind = nn.Upsample(size=(w, h), mode='nearest')(wind.view(-1, 36, w // 2, h // 2))
                hind = nn.Upsample(size=(w, h), mode='nearest')(hind.view(-1, 36, w // 2, h // 2))
                topk_p = nn.Upsample(size=(w,h), mode='nearest')(topk_p.view(-1,36, w // 2, h // 2))
                probs = nn.Softmax2d()(probs * topk_p)

                windt = wind.view(-1,w,h,1)
                hindt = hind.view(-1,w,h,1)
                windt = windt / float(w // 2.) - 1
                hindt = hindt / float(h // 2.) - 1
                sampler = torch.cat((windt, hindt), 3)
                with torch.no_grad():
                        imgs = front_images[0].unsqueeze(1).repeat(1,36,1,1,1).view(-1,img_c,256,256)
                        sampled_imgs = F.grid_sample(imgs, sampler)
                                
                        sampled_imgs = sampled_imgs.view(-1,36,img_c,w,h)
                
                gen_image = (sampled_imgs * probs.unsqueeze(2)).sum(dim=1)
                in_img = nn.Upsample(size=(w,h))(input_image)
                gen_loss += (gen_image - in_img).abs().sum(dim=1).view(b,-1).mean(dim=1).mean()

                # Get the topk indices
                topk = probs.topk(dim=1, k=9)
                wind = wind.gather(index=topk[1], dim=1)
                hind = hind.gather(index=topk[1], dim=1)
                index += 1
        
        loss_identity = torch.zeros((1,)).cuda()
        return gen_loss, loss_lowlevel, loss_identity, probs_lowlevel, gen_image, (sampler, probs, torch.zeros((b,1,256,256)).cuda()) 


def multiview_maskfaceloss_upscale(input_images, model, front_images, size=(256,256), compute_loss_identity=False):
        assert(len(front_images) == 1)
        b, c, w, h = input_image.size()

        # Predict the classes
        inp_emb = model.encoder(input_images[1])
        out_emb = model.encoder(front_images[0][1])

        emb = torch.cat((inp_emb, out_emb), 1)
        orig_probs = model.decoder[0:20](emb)
        probs_fine = model.decoder_finetune(orig_probs)

        # Calculate low-level probability
        probs = nn.Softmax2d()(probs_fine)
        b, c, w, h = probs.size()
        probs = probs.unsqueeze(1)
        fr_img = nn.Upsample(size=(w,h))(front_images[0])
        in_img = nn.Upsample(size=(w,h))(input_image)

        genimg_lowlevel = probs * fr_img.view(b,3,-1).unsqueeze(3).unsqueeze(4)
        genimg_lowlevel = genimg_lowlevel.sum(dim=2).view(b,3,w,h)
        loss_lowlevel = (genimg_lowlevel - in_img).abs().sum(dim=1).view(b,-1).mean(dim=1).mean()

        # Loss identity:
        if compute_loss_identity:
                loss_identity = nn.CrossEntropyLoss()(probs_fine.squeeze(), torch.linspace(0, w*h-1,w*h).long().unsqueeze(0).view(1,w,h).repeat(b,1,1).cuda())
        else:
                loss_identity = torch.zeros((1,))

        offset = probs.squeeze().max(dim=1)[1].view(b,1,w,h)
        wind = (offset % (w)).float() / (w / 2.) - 1
        hind = (offset / (w)).float() / (w / 2.) - 1
        offset = torch.cat((wind, hind), 1).float()
        if size[0] > w:
                # Then use this as the offset for doing the higher level one
                sampler = model.sampler_decoder(orig_probs)

                offset = nn.Upsample(size=size)(offset)
                gen_image = F.grid_sample(front_images[0][0], (offset + sampler).permute(0,2,3,1).contiguous())

                input_image = nn.Upsample(size=size)(input_images[0])
                gen_loss = (gen_image - input_image).abs().sum(dim=1).view(b,-1).mean(dim=1).mean()
        else:
                gen_loss = torch.zeros((1,)).cuda()
                sampler = torch.zeros(offset.size()).cuda()
                gen_image = genimg_lowlevel

        return gen_loss, loss_lowlevel, loss_identity, probs, gen_image 


def multiview_maskfaceloss_clusters(input_image, model, front_images):
        assert(len(front_images) == 1)
        b, c, w, h = input_image.size()

        # Predict the classes
        inp_emb = model.encoder(input_image)
        out_emb = model.encoder(front_images[0])

        emb = torch.cat((inp_emb, out_emb), 1)
        classes_inp = model.decoder(inp_emb)
        classes_fro = model.decoder(out_emb)

        # Get probability per class
        classes_inp = nn.Softmax2d()(classes_inp)
        classes_fro = nn.Softmax2d()(classes_fro)

        affine_decoders = model.affine_decoder(emb.squeeze()).view(b, -1, 2, 3).contiguous()

        # Add offset to affine trans
        offset = torch.eye(3)[0:2,:].unsqueeze(0).unsqueeze(0).cuda()
        affine_decoders = affine_decoders + offset
        

        # And then do transformations + probabilities
        trans_image = front_images[0].unsqueeze(1).repeat(1, affine_decoders.size(1), 1, 1, 1)
        trans_image = trans_image.view(-1,c,w,h) 
        affine_decoders = affine_decoders.view(-1, 2, 3).contiguous()
       
        affine_grid = F.affine_grid(affine_decoders, size=(affine_decoders.size(0), 3, w, h)) 
        gen_image = F.grid_sample(trans_image, affine_grid)
        
        loss_image = gen_image.view(b, -1, c, w, h).contiguous() - input_image.unsqueeze(1)
        loss_image = loss_image * classes_inp.unsqueeze(2)

        loss = loss_image.abs().sum(dim=2).view(b, -1, w * h).contiguous().mean(dim=2).sum(dim=1).mean()

        result_image = gen_image.view(b, -1, c, w, h) * classes_inp.unsqueeze(2)
        result_image = result_image.sum(dim=1).contiguous()

        # Get mean colors of loss
        mean_colors_inp = mean_colors_clusters(classes_inp, input_image)
        mean_colors_fro = mean_colors_clusters(classes_fro, front_images[0])
        
        # Loss 1: The clusters should have pixels of similar colors
        # Extract clusters from first image
        loss_cluster = cluster_loss(classes_inp, input_image, mean_colors_inp) + \
                       cluster_loss(classes_fro, front_images[0], mean_colors_fro)

        # Loss 2: The affine trans should pull similar clusters together
        mean_locs_inp = mean_locations_clusters(classes_inp)
        mean_locs_fro = mean_locations_clusters(classes_fro)

        loss_aff_cluster = dist_cluster_colloss(affine_decoders, mean_locs_inp, mean_colors_inp, result_image, classes_inp)

        return loss, loss_cluster, loss_aff_cluster, result_image.detach(), classes_inp

def dist_cluster_colloss(affine_grid, mean_locs_inp, mean_colors_inp, gen_image, classes_inp):
        # Cheat to make a xy grid of required size
        with torch.no_grad():
                offset = torch.eye(3)[0:2,:].unsqueeze(0).cuda().repeat(affine_grid.size(0), 1, 1)
                grid = F.affine_grid(offset, size=(affine_grid.size(0), 1, gen_image.size(2), gen_image.size(3))).squeeze().permute(0,3,1,2)
        grid_sampler = F.affine_grid(affine_grid, size=(affine_grid.size(0), 2, gen_image.size(2), gen_image.size(3)))
        
        # Where positions come from 
        grid_loc = F.grid_sample(grid, grid_sampler)
        b = gen_image.size(0)
        _, c, w, h = grid_loc.size()

        grid_sampler = (grid_loc.view(b, -1, c, w, h) * classes_inp.unsqueeze(2)).sum(dim=1)


        # Now minimize error based on location and nearest col
        dist = (gen_image.unsqueeze(1) - mean_colors_inp).pow(2).sum(dim=2)
        
        # Find class that is associated with the given color
        b = dist.size(0)
        class_id = dist.min(dim=1)[1].view(b, -1, 1).repeat(1,1,2)
        
        # Now move to nearest class
        locations = torch.gather(mean_locs_inp[:,:,0:2], dim=1, index=class_id)
        locations = locations.view(b, 256, 256, 2).contiguous().permute(0,3,1,2)

        err = (locations - grid_sampler).abs().sum(dim=3).sum(dim=2).sum(dim=1) / (gen_image.size(2) * gen_image.size(3))

        return err.mean()
        

def dist_cluster_loss(affine_decoders, mean_locs_inp, mean_locs_fro):
        # Reorganize to proper size
        mean_locs_inp = mean_locs_inp.view(mean_locs_inp.size(0)*mean_locs_inp.size(1), 3, 1)[:,0:2,:]
        mean_locs_fro = mean_locs_fro.view(mean_locs_fro.size(0)*mean_locs_fro.size(1), 3, 1)
        err = (mean_locs_inp  - affine_decoders.bmm(mean_locs_fro)).abs().sum(dim=1).mean()
        return err

def mean_locations_clusters(classes):
        locations = torch.ones((classes.size(0), classes.size(1), 3)).cuda()

        # Cheat to make a xy grid of required size
        offset = torch.eye(3)[0:2,:].unsqueeze(0).cuda().repeat(classes.size(0), 1, 1)
        grid = F.affine_grid(offset, size=(classes.size(0), 1, classes.size(2), classes.size(3))).squeeze().permute(0,3,1,2)

        for cluster in range(0, classes.size(1)):
                mean_loc = (classes[:,cluster:cluster+1,:,:] * grid).sum(dim=2).sum(dim=2) / \
                            classes[:,cluster:cluster+1,:,:].sum(dim=2).sum(dim=2)

                locations[:,cluster,0:2] = mean_loc

        return locations

def mean_colors_clusters(classes, image):
        mean_colors = torch.zeros((image.size(0), classes.size(1), 3, 1, 1)).cuda()

        for cluster in range(0, classes.size(1)):
                mean_col = (classes[:,cluster:cluster+1,:,:] * image).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True) / \
                            classes[:,cluster:cluster+1,:,:].sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
                mean_colors[:, cluster,:,:,:] = mean_col

        return mean_colors

def cluster_loss(clusters, image, mean_colors):
        loss = torch.zeros((1,)).cuda()
        for cluster in range(0, clusters.size(1)):
                tloss = clusters[:,cluster:cluster+1,:,:] * (image - mean_colors[:,cluster,:,:,:]).abs()
                tloss = tloss.sum(dim=1).sum(dim=1).sum(dim=1) / (image.size(2) * image.size(3))

                loss += tloss.mean()

        return loss / float(cluster)

