#!/usr/bin/env python
# coding=utf-8

from common_net import *

class COCOSharedDis(nn.Module):
    def __init__(self, params):
        super(COCOSharedDis, self).__init__()

        ch = params['ch']
        input_dim = params['input_dim']
        n_front_layer  = params['n_front_layer']
        n_shared_layer = params['n_shared_layer']
        n_domain       = params['domain_number']       
        print(params)

        embeddingnet_list = []
        for i in xrange(n_domain):
            model, tch = self._make_front_net(ch, input_dim, n_front_layer, n_shared_layer == 0)
            embeddingnet_list.append(model)
        self.model_As       = nn.ModuleList(embeddingnet_list)
        self.model_S        = self._make_shared_net(tch, n_shared_layer)

    def _make_front_net(self, ch, input_dim, n_layer, add_classifier_layer = False):
        model = []
        model += [LeakyReLUConv2d(input_dim, ch, kernel_size = 7, stride = 2, padding = 3)] 
        tch = ch
        for i in range(1, n_layer):
            model += [LeakyReLUConv2d(tch, tch * 2, kernel_size = 3, stride = 2, padding = 1)] 
            tch *= 2
        if add_classifier_layer:
            model += [nn.Conv2d(tch, 1, kernel_size = 1, stride = 1, padding = 0)]

        return nn.Sequential(*model), tch

    def _make_shared_net(self, ch, n_layer):
        model = []
        if n_layer == 0:
            return nn.Sequential(*model)
        tch = ch
        for i in range(0, n_layer):
            model += [LeakyReLUConv2d(tch, tch * 2, kernel_size = 3, stride = 2, padding = 1)]
            tch *= 2
        model += [nn.Conv2d(tch, 1, kernel_size = 1, stride = 1, padding = 0)]
        return nn.Sequential(*model)

    def cuda(self, gpu):
        self.model_As.cuda(gpu)
        self.model_S.cuda(gpu)

    def forward(self, x, a):
        #print('x.size() = ', x.size())
        out_A = self.model_S(self.model_As[a](x))
        #print('out_A.size = ', out_A.size())
        out_A = out_A.view(-1)
        outs_A = []
        outs_A.append(out_A)
        return outs_A

class COCOResGen(nn.Module):
    def __init__(self, params):
        super(COCOResGen, self).__init__()
        input_dim = params['input_dim']
        ch        = params['ch']
        n_enc_front_blk = params['n_enc_front_blk']
        n_enc_res_blk   = params['n_enc_res_blk']
        n_enc_shared_blk= params['n_enc_shared_blk']
        n_gen_shared_blk= params['n_gen_shared_blk']
        n_gen_res_blk   = params['n_gen_res_blk']
        n_gen_front_blk = params['n_gen_front_blk']
        n_domain        = params['domain_number']       
        print(params)

        # first part
        embeddingnet_enc_list = []
        for i in xrange(n_domain):
            enc = []
            enc += [LeakyReLUConv2d(input_dim, ch, kernel_size = 7, stride = 1, padding = 3)]
            tch = ch
            for j in xrange(1, n_enc_front_blk):
                enc += [ReLUINSConv2d(tch, tch * 2, kernel_size = 3, stride = 2, padding = 1)]
                tch *= 2
            for j in range(0, n_enc_res_blk):
                enc += [INSResBlock(tch, tch)]
            enc = nn.Sequential(*enc)
            embeddingnet_enc_list.append(enc)

        sch = tch

        # middle part, shared enc + dec
        enc_shared = []
        for i in range(0, n_enc_shared_blk):
            enc_shared += [INSResBlock(tch, tch)]
        enc_shared += [GaussianNoiseLayer()]

        dec_shared = []
        for i in range(0, n_gen_shared_blk):
            dec_shared += [INSResBlock(tch, tch)]

        # second part 
        embeddingnet_dec_list = []
        for i in xrange(n_domain):
            dec = []
            tch = sch
            for j in xrange(0, n_gen_res_blk):
                dec += [INSResBlock(tch, tch)]
            for j in xrange(0, n_gen_front_blk - 1):
                dec += [ReLUINSConvTranspose2d(tch, tch // 2, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)]
                tch = tch // 2
            dec += [nn.ConvTranspose2d(tch, input_dim, kernel_size = 1, stride = 1, padding = 0)]
            dec += [nn.Tanh()]
            dec = nn.Sequential(*dec)
            embeddingnet_dec_list.append(dec)

        self.encodes = nn.ModuleList(embeddingnet_enc_list)
        self.enc_shared = nn.Sequential(*enc_shared)
        self.dec_shared = nn.Sequential(*dec_shared)
        self.decodes = nn.ModuleList(embeddingnet_dec_list)

    def forward(self, x, a, b):
        out = self.encodes[a](x)
        shared = self.enc_shared(out)
        out    = self.dec_shared(shared)
        x_ab   = self.decodes[b](out)
        return x_ab, shared

class COCOResGen2(nn.Module):
    def __init__(self, params):
        super(COCOResGen2, self).__init__()
        input_dim = params['input_dim']
        ch        = params['ch']
        n_enc_front_blk = params['n_enc_front_blk']
        n_enc_res_blk   = params['n_enc_res_blk']
        n_enc_shared_blk= params['n_enc_shared_blk']
        n_gen_shared_blk= params['n_gen_shared_blk']
        n_gen_res_blk   = params['n_gen_res_blk']
        n_gen_front_blk = params['n_gen_front_blk']
        n_domain        = params['domain_number']       
        self.domain_in  = params['domain_in']
        print(params)

        # first part
        embeddingnet_enc_list = []
        for i in xrange(self.domain_in):
            enc = []
            enc += [LeakyReLUConv2d(input_dim, ch, kernel_size = 7, stride = 1, padding = 3)]
            tch = ch
            for j in xrange(1, n_enc_front_blk):
                enc += [LeakyReLUConv2d(tch, tch * 2, kernel_size = 3, stride = 2, padding = 1)]
                tch *= 2
            for j in range(0, n_enc_res_blk):
                enc += [INSResBlock(tch, tch)]
            enc = nn.Sequential(*enc)
            embeddingnet_enc_list.append(enc)

        sch = tch

        # middle part, shared enc + dec
        enc_shared = []
        for i in range(0, n_enc_shared_blk):
            enc_shared += [INSResBlock(tch, tch)]
        enc_shared += [GaussianNoiseLayer()]

        dec_shared = []
        for i in range(0, n_gen_shared_blk):
            dec_shared += [INSResBlock(tch, tch)]

        # second part 
        embeddingnet_dec_list = []
        for i in xrange(n_domain):
            dec = []
            tch = sch
            for j in xrange(0, n_gen_res_blk):
                dec += [INSResBlock(tch, tch)]
            for j in xrange(0, n_gen_front_blk - 1):
                dec += [LeakyReLUConvTranspose2d(tch, tch // 2, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)]
                tch = tch // 2
            dec += [nn.ConvTranspose2d(tch, input_dim, kernel_size = 1, stride = 1, padding = 0)]
            dec += [nn.Tanh()]
            dec = nn.Sequential(*dec)
            embeddingnet_dec_list.append(dec)

        self.encodes = nn.ModuleList(embeddingnet_enc_list)
        self.enc_shared = nn.Sequential(*enc_shared)
        self.dec_shared = nn.Sequential(*dec_shared)
        self.decodes = nn.ModuleList(embeddingnet_dec_list)

    def forward(self, x, a, b):
        # add for first shared
        a = a % self.domain_in

        out = self.encodes[a](x)
        shared = self.enc_shared(out)
        out    = self.dec_shared(shared)
        x_ab   = self.decodes[b](out)
        return x_ab, shared
