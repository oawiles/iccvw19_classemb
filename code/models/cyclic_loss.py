import torch.nn as nn
import torch.nn.functional as F
import torch

EPS=1e-4

class CyclicLoss(nn.Module):
        def __init__(self):
                super(CyclicLoss, self).__init__()
        
        def forward(self, prob_ce12, prob_ce21):
                prob_ce12 = prob_ce12.squeeze()
                prob_ce21 = prob_ce21.squeeze()
                b, c, _, _ = prob_ce12.size()
                prob12 = prob_ce12.permute(0,2,3,1).view(b, -1, c)
                prob21 = prob_ce21.permute(0,2,3,1).view(b, -1, c)

                # Renormalize the probabilities : so that can't map everything to the head
                prob12 = torch.min((prob12.clamp(min=EPS) / prob12.sum(dim=1, keepdim=True).clamp(min=EPS)), (prob12.clamp(min=EPS) / prob12.sum(dim=2, keepdim=True).clamp(min=EPS)))
                prob21 = torch.min((prob21.clamp(min=EPS) / prob21.sum(dim=1, keepdim=True).clamp(min=EPS)), (prob21.clamp(min=EPS) / prob21.sum(dim=2, keepdim=True).clamp(min=EPS)))
                
                probs = (prob12 * prob21.permute(0,2,1)).sum(dim=2).clamp(min=0, max=1)
                probs = nn.BCELoss()(probs, torch.ones(probs.size()).cuda())
                return probs * 0.1, prob_ce12, prob_ce21

class CyclicLossOld(nn.Module):
        def __init__(self):
                super(CyclicLossOld, self).__init__()
        
        def forward(self, prob_ce12, prob_ce21):
                prob_ce12 = prob_ce12.squeeze()
                prob_ce21 = prob_ce21.squeeze()
                b, c, _, _ = prob_ce12.size()
                prob12 = prob_ce12.permute(0,2,3,1).view(b, -1, c)
                prob21 = prob_ce21.permute(0,2,3,1).view(b, -1, c)

                # Renormalize the probabilities : so that can't map everything to the head
                err = ((prob12.sum(dim=1) - 1).abs().mean(dim=1) + (prob21.sum(dim=1) - 1).abs().mean(dim=1))
                err = err.abs().mean()               
 
                probs = (prob12 * prob21.permute(0,2,1)).sum(dim=2).clamp(min=0, max=1)
                probs = nn.BCELoss()(probs, torch.ones(probs.size()).cuda())
                #probs = (1 - probs).view(b, -1).mean(dim=1).mean()
                #print(probs, err)
                return 0.05 * probs + err * 0.5, prob_ce12, prob_ce21

class CyclicLossSamplers(nn.Module):
        def __init__(self):
                super(CyclicLossSamplers, self).__init__()
                self.identity = F.affine_grid(torch.eye(3)[0:2,:].unsqueeze(0), size=torch.Size((1,1,32,32))).cuda()
 
        def forward(self, cyc1, cyc2):
                cyc21 = F.grid_sample(cyc1.permute(0,3,1,2), cyc2)

                identity = self.identity.permute(0,3,1,2).repeat(cyc1.size(0), 1, 1, 1)
                err = (cyc21 - identity).abs().sum(dim=1).view(cyc1.size(0), -1).mean(dim=1).mean()
                return err, None, None

#class CyclicLoss(nn.Module):
#        def __init__(self):
#                super(CyclicLoss, self).__init__()
#               
#        def forward(self, prob_ce12, prob_ce21):
#               prob_ce12 = prob_ce12.squeeze()
#                prob_ce21 = prob_ce21.squeeze()
#                b, c, _, _ = prob_ce12.size()

#                prob12 = prob_ce12.permute(0,2,3,1).view(b,-1,c)
#                prob21 = prob_ce21.permute(0,2,3,1).view(b,-1,c)

#                prob12 = prob12.pow(3) / (prob12.max(dim=1, keepdim=True)[0] * prob12.max(dim=2, keepdim=True)[0]).clamp(min=1e-4)
#                prob21 = prob21.pow(3) / (prob21.max(dim=1, keepdim=True)[0] * prob21.max(dim=2, keepdim=True)[0]).clamp(min=1e-4)

#                probs = (prob12 * prob21.permute(0,2,1)).sum(dim=2).clamp(min=0, max=1)
#                probs = nn.BCELoss()(probs, torch.ones(probs.size()).cuda())
#                return probs, prob_ce12, prob_ce21

def compare_images(gen_image, gt_image):
        b, c, w, h = gen_image.size()

        gt_image = nn.Upsample(size=(w,h))(gt_image)

        err = (gen_image - gt_image).abs().sum(dim=1).view(b, -1).mean(dim=1).mean()
        return err

def multiview_maskfaceloss_cyclic(input_image, model, front_images, size=(256,256), compute_loss_identity=False):
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
        return genimg_lowlevel, probs
