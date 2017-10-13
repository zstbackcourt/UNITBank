#!/usr/bin/env python
# coding=utf-8

from cocogan_nets import *
from init import *
from helpers import get_model_list #, _compute_fake_acc, _compute_true_acc
import torch
import torch.nn as nn
import os
import itertools

class COCOGANTrainer(nn.Module):
    def __init__(self, hyperparameters):
        super(COCOGANTrainer, self).__init__()
        lr = hyperparameters['lr']
        # init networks
        cmd = 'self.dis = %s(hyperparameters[\'dis\'])' % hyperparameters['dis']['name']
        exec(cmd)
        print(cmd)
        cmd = 'self.gen = %s(hyperparameters[\'gen\'])' % hyperparameters['gen']['name']
        exec(cmd)
        print(cmd)

        # setup the optimizers
        self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr = lr, betas = (0.5, 0.999), weight_decay = 0.0001)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr = lr, betas = (0.5, 0.999), weight_decay = 0.0001)

        # weight initialization
        self.dis.apply(gaussian_weights_init)
        self.gen.apply(gaussian_weights_init)

        # setup the loss function for training
        self.L1Loss = torch.nn.L1Loss()

    def _compute_kl(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def gen_update(self, images_a, images_b, hyperparameters, a, b):
        self.gen.zero_grad()
        x_aa, shared_aa   = self.gen(images_a, a, a)
        x_ab, shared_ab   = self.gen(images_a, a, b)
        x_aba, shared_aba = self.gen(x_ab, b, a)
        outs_b            = self.dis(x_ab, b)
        for it, (out_b, ) in enumerate(itertools.izip(outs_b)):
            outputs_b     = nn.functional.sigmoid(out_b)
            all_ones      = Variable(torch.ones(outputs_b.size(0))).cuda(self.gpu)
            if it == 0:
                ad_loss_b = nn.functional.binary_cross_entropy(outputs_b, all_ones)
            else:
                ad_loss_b += nn.functional.binary_cross_entropy(outputs_b, all_ones)

        enc_aa_loss  = self._compute_kl(shared_aa)
        enc_ab_loss  = self._compute_kl(shared_ab)
        enc_aba_loss = self._compute_kl(shared_aba)

        ll_loss_aba  = self.L1Loss(x_aba, images_a)
        ll_loss_aa   = self.L1Loss(x_aa,  images_a)

        total_loss = hyperparameters['gan_w'] * (ad_loss_b) + \
                     hyperparameters['ll_direct_link_w'] * (ll_loss_aa) + \
                     hyperparameters['ll_cycle_link_w'] * (ll_loss_aba) + \
                     hyperparameters['kl_direct_link_w'] * (enc_aa_loss + enc_ab_loss) + \
                     hyperparameters['kl_cycle_link_w'] * (enc_aba_loss)
        total_loss.backward()
        self.gen_opt.step()
        self.gen_enc_aa_loss  = enc_aa_loss.data.cpu().numpy()[0]
        self.gen_enc_ab_loss  = enc_ab_loss.data.cpu().numpy()[0]
        self.gen_enc_aba_loss = enc_aba_loss.data.cpu().numpy()[0]
        self.gen_ad_loss_b    = ad_loss_b.data.cpu().numpy()[0]
        self.gen_ll_loss_aa   = ll_loss_aa.data.cpu().numpy()[0]
        self.gen_ll_loss_aba  = ll_loss_aba.data.cpu().numpy()[0]
        self.gen_total_loss   = total_loss.data.cpu().numpy()[0]
        return x_ab, x_aba

    def dis_update(self, images_a, images_b, hyperparameters, a, b):
        self.dis.zero_grad()
        x_ab, shared      = self.gen(images_a, a, b)
        data_b            = torch.cat((images_b, x_ab), 0)
        res_b             = self.dis(data_b, b)

        for it, (this_b, ) in enumerate(itertools.izip(res_b)):
            #print(this_b)
            out_b         = nn.functional.sigmoid(this_b)
            out_true_b, out_fake_b = torch.split(out_b, out_b.size(0) // 2, 0)
            out_true_n    = out_true_b.size(0)
            out_fake_n    = out_fake_b.size(0)
            all1          = Variable(torch.ones((out_true_n)).cuda(self.gpu))
            all0          = Variable(torch.zeros((out_fake_n)).cuda(self.gpu))
            ad_true_loss_b = nn.functional.binary_cross_entropy(out_true_b, all1)
            ad_fake_loss_b = nn.functional.binary_cross_entropy(out_fake_b, all0)
            if it == 0:
                ad_loss_b  = ad_true_loss_b + ad_fake_loss_b
            else:
                ad_loss_b += ad_true_loss_b + ad_fake_loss_b

        loss = hyperparameters['gan_w'] * (ad_loss_b)
        loss.backward()
        self.dis_opt.step()
        self.dis_loss = loss.data.cpu().numpy()[0]
        return

    def assemble_outputs(self, images_a, images_b, network_outputs):
        images_a = self.normalize_image(images_a)
        images_b = self.normalize_image(images_b)
        x_ab     = self.normalize_image(network_outputs[0])
        x_aba    = self.normalize_image(network_outputs[1])
        return torch.cat((images_a[0:1, ::], x_ab[0:1, ::], x_aba[0:1, ::], images_b[0:1, ::]), 3)

    def resume(self, snapshot_prefix):
        dirname = os.path.dirname(snapshot_prefix)
        last_model_name = get_model_list(dirname, "gen")
        if last_model_name is None:
            return 0
        self.gen.load_state_dict(torch.load(last_model_name))
        iterations = int(last_model_name[-12: -4])
        last_model_name = get_model_list(dirname, "dis")
        self.dis.load_state_dict(torch.load(last_model_name))
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_prefix, iterations):
        gen_filename = "%s_gen_%08d.pkl" % (snapshot_prefix, iterations + 1)
        dis_filename = "%s_dis_%08d.pkl" % (snapshot_prefix, iterations + 1)
        torch.save(self.gen.state_dict(), gen_filename)
        torch.save(self.dis.state_dict(), dis_filename)

    def cuda(self, gpu):
        self.gpu = gpu
        self.dis.cuda(gpu)
        self.gen.cuda(gpu)

    def normalize_image(self, x):
        return x[:, 0: 3, :, :]

