# Add an encoder/decoder
from tensorboardX import SummaryWriter

from utils_scheduler.TrackLoss import TrackLoss
import os
import numpy as np

from models_multiview import FrontaliseModelMasks_wider_hierarchy, FrontaliseModelMasks_wider_hierarchy64

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse

BASE_LOCATION = os.environ['BASE_LOCATION']


arguments = argparse.ArgumentParser()
arguments.add_argument('--lr', type=float, default=0.01)
arguments.add_argument('--lamb', type=float, default=1)
arguments.add_argument('--momentum', type=float, default=0.9)
arguments.add_argument('--load_old_model', action='store_true', default=False)
arguments.add_argument('--num_views', type=int, default=2, help='Number of source views + 1 (e.g. the target view) so set = 2 for 1 source view')
arguments.add_argument('--continue_epoch', type=int, default=0)
arguments.add_argument('--crop_size', type=int, default=180)
arguments.add_argument('--num_additional_ids', type=int, default=32)
arguments.add_argument('--use_identityloss', action='store_true', default=False)
arguments.add_argument('--num_workers', type=int, default=10)
arguments.add_argument('--max_percentile', type=float, default=0.83)
arguments.add_argument('--diff_percentile', type=float, default=0.1)
arguments.add_argument('--batch_size', type=int, default=16)
arguments.add_argument('--num_outer_samples', type=int, default=36)
arguments.add_argument('--use_cyclic', type=str)
arguments.add_argument('--use_ab', action='store_true')
arguments.add_argument('--use_cyclic_other', action='store_true')
arguments.add_argument('--use_confidence', type=str, default='True')
arguments.add_argument('--upsample_type', type=str, default='hierarchy')
arguments.add_argument('--use_curriculum', action='store_true')
arguments.add_argument('--log_dir', type=str, default=BASE_LOCATION+'/logging/iccv19_selfsup/runs/')
arguments.add_argument('--embedding_size', type=int, default=256)
arguments.add_argument('--run_dir', type=str, default='upscalevox2%d_fabnet/finetune_id%slr_%.4f_lambda%.4f_nv%d_addids%d_cropsize%d_data%s_curr%s_cycnewloss%s_ups%s_conf%s_ab%s')
arguments.add_argument('--old_model', type=str, default=BASE_LOCATION + '')
arguments.add_argument('--dataset', type=str, default='voxceleb2')
arguments.add_argument('--model_epoch_path', type=str, default=BASE_LOCATION + '/logging/iccv19_selfsup/models/disentangling/finetuneid%s_upscalevox2%.4f_emb%d_bs%d_lambda%.4f_nv%d_addids%d_cropsize%d_data%s_curr%s_cycnewloss%s_ups%s_conf%s_ab%s')
opt = arguments.parse_args()


opt.run_dir = opt.run_dir % (opt.embedding_size, opt.use_identityloss, opt.lr, opt.lamb, opt.num_views, opt.num_additional_ids, opt.crop_size, 
                                opt.dataset, opt.use_curriculum, opt.use_cyclic, opt.upsample_type, opt.use_confidence, opt.use_ab)
print(opt.run_dir)
opt.model_epoch_path = opt.model_epoch_path % (opt.use_identityloss, opt.lr, opt.embedding_size, opt.batch_size, opt.lamb, 
                                opt.num_views, opt.num_additional_ids, opt.crop_size,  opt.dataset, opt.use_curriculum, opt.use_cyclic, opt.upsample_type, 
                                opt.use_confidence, opt.use_ab)
opt.model_epoch_path = opt.model_epoch_path + 'epoch%s.pth'

opt.sep_dec_conf = opt.use_confidence == 'TrueSeperate'
opt.use_confidence = 'True' in opt.use_confidence

print('Confidence : ', opt.use_confidence, ' with sep dec ', opt.sep_dec_conf)

opt.use_cyclic_old = ('old' in opt.use_cyclic and 'True' in opt.use_cyclic)
opt.use_cyclic = 'True' in opt.use_cyclic
print("Cyclic : ", opt.use_cyclic, ' old one? ', opt.use_cyclic_old)

if opt.use_cyclic_old:
        from models.cyclic_loss import CyclicLossOld as CyclicLoss
else:
        from models.cyclic_loss import CyclicLoss

if opt.use_curriculum:
        if opt.upsample_type == 'hierarchy64':
                opt.size = 64
        else:
                opt.size = 32
        opt.frames = 50
else:
        opt.frames=50
        opt.size = 128
print(opt.size)

if opt.use_ab:
        num_inputs = 2
else:
        num_inputs = 3

print(num_inputs)
if opt.upsample_type == 'hierarchy64':
        model = FrontaliseModelMasks_wider_hierarchy64(input_nc=num_inputs, inner_nc=opt.embedding_size, num_outer_samples=opt.num_outer_samples, 
                        num_masks=0, num_additional_ids=opt.num_additional_ids, sep_dec_conf=opt.sep_dec_conf, 
                        num_super_pixels=32, use_confidence=opt.use_confidence, tracking=(opt.dataset == 'kinetics'))
elif opt.upsample_type == 'hierarchy':
        model = FrontaliseModelMasks_wider_hierarchy(input_nc=num_inputs, inner_nc=opt.embedding_size, num_outer_samples=opt.num_outer_samples,
                        num_masks=0, num_additional_ids=opt.num_additional_ids, sep_dec_conf=opt.sep_dec_conf,
                        num_super_pixels=32, use_confidence=opt.use_confidence, tracking=(opt.dataset == 'kinetics'))
else:
        model = FrontaliseModelMasks_wider_hierarchy(input_nc=num_inputs, inner_nc=opt.embedding_size, sep_dec_conf=opt.sep_dec_conf,
                        num_masks=0, num_additional_ids=opt.num_additional_ids, num_super_pixels=2, use_confidence=opt.use_confidence)

model.lr = opt.lr
model.momentum = opt.momentum
writer = SummaryWriter('/%s/%s' % (opt.log_dir, opt.run_dir))

model = model.cuda()

criterion_reconstruction = nn.L1Loss(reduce=False).cuda()

print("Choosing loss function...")
if 'hierarchy' in opt.upsample_type and opt.use_confidence:
        from loss_functions import multiview_hierarchy_confidence as loss_upsample
elif 'hierarchy' in opt.upsample_type:
        from loss_functions import multiview_hierarchy as loss_upsample
else:
        from loss_functions import multiview_maskfaceloss_upscale as loss_upsample

print("Loading dataset...")
if opt.dataset == 'pennaction_propsplit':
        print("Loading penn action")
        from PennAction import PennAction  as Dataset
        opt.frames = 20
elif opt.dataset == 'horses':
        print("Loading horses")
        from Horses_indrange import Horses as Dataset
        opt.frames = 50 

elif opt.dataset == 'pennaction':
        print("Loading penn action")
        from PennAction import PennAction  as Dataset
        opt.frames = 20
elif opt.dataset == 'bbcpose_small_jitter':
        print("Loading bbcpose small with jitter...")
        from BBCPose import BBCPoseAnnotationsJitter as Dataset
elif opt.dataset == 'bbcpose_small':
        print("Loading bbcpose...")
        from BBCPose import BBCPoseAnnotations as Dataset
elif opt.dataset == 'bbcposecentered':
        print("Loading bbccenteredpose...")
        from BBCPose import BBCPoseAnnotationsCentered as Dataset
elif opt.dataset == 'kinetics':
        print("kinetics")
        from Kinetics import Kinetics as Dataset
elif opt.dataset == 'animals':
        print("Loading animals...")
        from animals import Animals as Dataset 
elif opt.dataset == 'wild3d':
        print("Loading poses in the wild...")
        opt.frames=20
        from Pose3DWild import Pose3DWild as Dataset 
elif opt.dataset == 'humans36m':
        print("Loading humans...")
        opt.frames=20
        from Humans36M import Humans36M as Dataset 
else:
        print("Loading vox2...")
        from VoxCelebData_withmask import VoxCeleb2 as Dataset

optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)



def train(epoch, model, criterion, optimizer, num_additional_ids=5, minpercentile=0, maxpercentile=50):
        train_set = Dataset(opt.num_views, epoch, 1, jittering=True, frames=opt.frames, use_ab=opt.use_ab)
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.num_workers, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    
        t_loss = 0
        c_loss = 0
        a_loss = 0

        for iteration, batch in enumerate(training_data_loader, 1):
                optimizer.zero_grad()

                if opt.dataset == 'bbcpose_small_jitter' or opt.dataset == 'bbcposecentered':
                        input_image = Variable(batch['face1'][0][0]).cuda()
                        front_images = [Variable(batch['face1'][0][1]).cuda()]

                        input_imagesjit = Variable(batch['face1'][1][0]).cuda()
                        front_imagesjit = [Variable(batch['face1'][1][1]).cuda()]
                else:
                        input_image = Variable(batch['face1'][0]).cuda()
                        front_images = [Variable(batch['face1'][1]).cuda()]
                        input_imagejit = Variable(batch['face1'][0]).cuda()
                        front_imagesjit = [Variable(batch['face1'][1]).cuda()]

                output_image = front_images[0]
                loss, closs, _, probabilities, output_image, conf = loss_upsample([input_image, input_imagesjit], model, [front_images, front_imagesjit], size=(opt.size, opt.size), tracking=(opt.dataset == 'kinetics'))

                if opt.use_identityloss:                
                        _, _, aloss, _, _, _ = loss_upsample(input_image, model, [input_image], size=(opt.size, opt.size), compute_loss_identity=True, tracking=(opt.dataset == 'kinetics'))
                        (loss + closs + aloss * 0.01).backward()
                elif opt.use_cyclic:
                        _, _, _, probabilities21, _, _ = loss_upsample([front_images[0], front_imagesjit[0]], model, [[input_image], [input_imagesjit]], size=(32,32), tracking=(opt.dataset == 'kinetics')) # Needs to be 64 when also doing the upscaling
                        probabilities = probabilities.clamp(min=0, max=1)
                        probabilities21 = probabilities21.clamp(min=0, max=1)
                        aloss, prob1, prob2 = CyclicLoss()(probabilities, probabilities21)
                        #(loss + closs + aloss * opt.size / 32. * opt.lamb).backward()
                        (loss + closs + aloss * opt.lamb).backward()
                else:
                        aloss = torch.zeros((1,)).cuda()
                        (loss + closs).backward()

                for p in model.parameters():
                        if p.data.grad:
                                if not p.data.grad.min() == p.data.grad.min():
                                        import pdb
                                        pdb.set_trace()
                                p.data.grad.clamp(min=-1,max=1)

                optimizer.step()


                t_loss += loss.cpu().item()
                c_loss += closs.cpu().item()
                a_loss += aloss.cpu().item()
                if iteration == 1:
                        # Convert to lab space if necessary
                        if opt.use_ab:
                                front_images[0] = convert_ab2rgb(front_images[0].detach())
                                input_image = convert_ab2rgb(input_image.detach())
                                output_image = convert_ab2rgb(output_image.detach())
                                writer.add_image('Image_train/%d_input' % iteration, input_image, epoch)
                                writer.add_image('Image_train/%d_front' % iteration, front_images[0], epoch)
                                writer.add_image('Image_train/%d_output' % iteration, output_image, epoch)

                        else:

                                writer.add_image('Image_train/%d_input' % iteration, torchvision.utils.make_grid(input_image[0:8,:,:,:].data), epoch)
                                writer.add_image('Image_train/%d_front' % iteration, torchvision.utils.make_grid(front_images[0][0:8,:,:,:].data), epoch)
                                writer.add_image('Image_train/%d_output' % iteration, torchvision.utils.make_grid(output_image[0:8,:,:,:].data), epoch)
                        if opt.use_cyclic:
                                b, c, _, _ = prob1.size()
                                prob1 = prob1.view(b, c, -1).unsqueeze(1)
                                writer.add_image('Image_train/%d_cyclic' % iteration, torchvision.utils.make_grid(prob1[0:8,:,:,:].data), epoch)
                        if opt.use_confidence:
                                writer.add_image('Image_train/%d_conf' % iteration, torchvision.utils.make_grid(conf[2][0:8,:,:,:].data, normalize=True), epoch) 


                if iteration % 10 == 0:
                        print("Train: Epoch {}: {}/{} with error {:.4f} | {:.4f} | {:.4f}". \
                                format(epoch, iteration, len(training_data_loader), t_loss / float(iteration), c_loss / float(iteration), a_loss / float(iteration)))
                if iteration == 150:
                        break

        return {'reconstruction_error' : t_loss / float(iteration), 'conf_error' : c_loss / float(iteration),
                'affine_error' : a_loss / float(iteration)}

def val(epoch, model, criterion, optimizer, minpercentile=0, maxpercentile=50):
        val_set = Dataset(opt.num_views, 0, 2, jittering=True, frames=opt.frames, use_ab=opt.use_ab) 

        val_data_loader = DataLoader(dataset=val_set, num_workers=opt.num_workers, batch_size=opt.batch_size, shuffle=False, drop_last=True)
    
        t_loss = 0
        c_loss = 0
        a_loss = 0

        for iteration, batch in enumerate(val_data_loader, 1):
                if opt.dataset == 'bbcpose_small_jitter' or opt.dataset == 'bbcposecentered':
                        input_image = Variable(batch['face1'][0][0]).cuda()
                        front_images = [Variable(batch['face1'][0][1]).cuda()]

                        input_imagesjit = Variable(batch['face1'][1][0]).cuda()
                        front_imagesjit = [Variable(batch['face1'][1][1]).cuda()]
                else:
                        input_image = Variable(batch['face1'][0]).cuda()
                        front_images = [Variable(batch['face1'][1]).cuda()]
                        input_imagejit = Variable(batch['face1'][0]).cuda()
                        front_imagesjit = [Variable(batch['face1'][1]).cuda()]

                offset = 1

                output_image = front_images[0]
                loss, closs, _, probabilities, output_image, conf = loss_upsample([input_image, input_imagesjit], model, [front_images, front_imagesjit], size=(opt.size, opt.size), tracking=(opt.dataset == 'kinetics'))

                t_loss += loss.cpu().item()
                c_loss += closs.cpu().item()
                if iteration % 1000 == 0 or iteration == 1:
                        if opt.use_ab:
                                front_images[0] = convert_ab2rgb(front_images[0].detach())
                                input_image = convert_ab2rgb(input_image.detach())
                                output_image = convert_ab2rgb(output_image.detach())
                                writer.add_image('Image_val/%d_input' % iteration, input_image, epoch)
                                writer.add_image('Image_val/%d_front' % iteration, front_images[0], epoch)
                                writer.add_image('Image_val/%d_output' % iteration, output_image, epoch)

                        else:

                                writer.add_image('Image_val/%d_input' % iteration, torchvision.utils.make_grid(input_image[0:8,:,:,:].data), epoch)
                                writer.add_image('Image_val/%d_front' % iteration, torchvision.utils.make_grid(front_images[0][0:8,:,:,:].data), epoch)
                                writer.add_image('Image_val/%d_output' % iteration, torchvision.utils.make_grid(output_image[0:8,:,:,:].data), epoch)
                        if opt.use_confidence:
                                writer.add_image('Image_val/%d_conf' % iteration, torchvision.utils.make_grid(conf[2][0:8,:,:,:].data, normalize=True), epoch) 
                        
                if iteration % 10 == 0:
                        print("Val: Epoch {}: {}/{} with error {:.4f} | {:.4f} | {:.4f}".format(epoch, iteration,
                                len(val_data_loader), t_loss / float(iteration), c_loss / float(iteration), a_loss / float(iteration)))

                if iteration == 40:
                        break

        return {'reconstruction_error' : t_loss / float(iteration), 'conf_error' : c_loss / float(iteration),
                'affine_error' : a_loss / float(iteration)}

def checkpoint(model, save_path, plateauscheduler):
        checkpoint_state = {'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict(), 'epoch' : model.epoch, 
                                                        'lr' : model.lr, 'momentum' : model.momentum, 'opts' : opt, 'scheduler' : plateauscheduler.state_dict()}

        torch.save(checkpoint_state, save_path)

def run():
        scheduler = TrackLoss()
        plateauscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        if opt.continue_epoch > 0:
                print("HERE")
                past_state = torch.load(opt.model_epoch_path % str(opt.continue_epoch - 1))
                model.load_state_dict(torch.load(opt.model_epoch_path % str(opt.continue_epoch - 1))['state_dict'])
                optimizer.load_state_dict(torch.load(opt.model_epoch_path % str(opt.continue_epoch - 1))['optimizer'])
                plateauscheduler.load_state_dict(torch.load(opt.model_epoch_path % str(opt.continue_epoch - 1))['scheduler'])

                opt.size = past_state['opts'].size

        for epoch in range(opt.continue_epoch, 10000):
                print(epoch, opt.size, opt.frames, scheduler.best_epoch)
                model.epoch = epoch
                model.optimizer_state = optimizer.state_dict()
                model.train()
                train_loss = train(epoch, model, criterion_reconstruction, optimizer)
                model.eval()
                with torch.no_grad():
                        loss = val(epoch, model, criterion_reconstruction, optimizer)

                scheduler.update(loss['reconstruction_error'] + loss['conf_error'], epoch)
                if scheduler.drop_learning_rate(epoch):
                        if opt.frames < 16:
                                opt.frames += 2
                        elif opt.size <= 100:
                                #opt.frames = 4
                                print(scheduler.best_epoch, scheduler.last_warm, opt.size)
                                checkpoint(model, opt.model_epoch_path % ('_curriculum' + str(opt.size)), plateauscheduler)
                                opt.size = opt.size * 2
                                if opt.size == 64:      
                                        offset = 1
                                elif opt.size == 128:
                                        offset = 2
                                else:
                                        offset = 3

                                scheduler = TrackLoss()
                        else:
                                plateauscheduler.step(loss['reconstruction_error'] + loss['conf_error'])

                writer.add_scalars('loss_recon/train_val', {'train' : train_loss['reconstruction_error'], 'val' : loss['reconstruction_error']}, epoch)
                writer.add_scalars('loss_conf/train_val', {'train' : train_loss['conf_error'], 'val' : loss['conf_error']}, epoch)
                writer.add_scalars('loss_affine/train_val', {'train' : train_loss['affine_error'], 'val' : loss['affine_error']}, epoch)

                if epoch % 10 == 0:
                        checkpoint(model, opt.model_epoch_path % str(epoch), plateauscheduler)

                        for i in range(1,15):
                                if os.path.exists(opt.model_epoch_path % str((epoch - i))):
                                        os.remove(opt.model_epoch_path % str((epoch - i)))



if __name__ == '__main__':
        if opt.load_old_model:
                opts = torch.load(opt.old_model)['opts']
                state_dict = model.state_dict()
                pretrained_dict = (torch.load(opt.old_model)['state_dict'])
                state_dict.update({k: v for k, v in pretrained_dict.items() if k in state_dict.items()})        
                model.load_state_dict(pretrained_dict)

                opt = opts
                opt.size = opt.size*2
                run()
        
        else:
                run()





