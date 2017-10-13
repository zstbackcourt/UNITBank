#!/usr/bin/env python
# coding=utf-8

import sys
from tools import *
from trainers import *
from datasets import *
import torchvision
import itertools
from common import *
import tensorboard
from tensorboard import summary
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--gpu', type = int, help = 'gpu id', default = 0)
parser.add_option('--resume', type = int, help = 'resume training ?', default = 0)
parser.add_option('--config', type = str, help = 'net configuration')
parser.add_option('--log', type = str, help = 'log path')
MAX_EPOCHS = 100000

def main(argv):
    (opts, args) = parser.parse_args(argv)

    # Load experiment setting
    assert isinstance(opts, object)
    config = NetConfig(opts.config)

    batch_size     = config.hyperparameters['batch_size']
    max_iterations = config.hyperparameters['max_iterations']

    # multi-domain loaders
    train_loaders  = []
    for i, train_x in enumerate(config.datasets.keys()):
        print('Domain %d = %s' % (i, train_x))
        train_loader = get_data_loader(config.datasets[train_x], batch_size)
        train_loaders.append(train_loader)

    # exec initialization of trainer
    trainer = []
    exec ('trainer = %s(config.hyperparameters)' % config.hyperparameters['trainer'])

    iterations = 0
    if opts.resume == 1:
        iterations = trainer.resume(config.snapshot_prefix)
    trainer.cuda(opts.gpu)

    ###### setup logger and repare image outputs
    train_writer = tensorboard.FileWriter("%s/%s" % (opts.log, os.path.splitext(os.path.basename(opts.config))[0]))
    image_directory, snapshot_directory = prepare_snapshot_and_image_folder(config.snapshot_prefix, iterations, config.image_save_iterations)

    domain_number = len(train_loaders)
    for ep in range(0, MAX_EPOCHS):
        for it, images in enumerate(itertools.izip(*train_loaders)):
            images_list = []
            for image in images:
                im = Variable(image.cuda(opts.gpu))
                images_list.append(im)
                #print('im shape = ', im.size())

            assembled_list = []
            for i in xrange(domain_number):
                for j in xrange(domain_number):
                    # first:  all of them VAE pass
                    if i == j:
                        continue
                        #trainer.vae_update(images_list[i], images_list[j], config.hyperparameters, i, j)
                    # second: all crossing pairs for GAN, let the lambda judge the 
                    else: # i != j
                        trainer.dis_update(images_list[i], images_list[j], config.hyperparameters, i, j)
                        image_outputs = trainer.gen_update(images_list[i], images_list[j], config.hyperparameters, i, j)

                        assembled = trainer.assemble_outputs(images_list[i], images_list[j], image_outputs)
                        assembled_list.append(assembled)

            assembled_images = torch.cat(assembled_list, 2)
            # Dump training stats in log file

            for t in xrange(domain_number * domain_number - domain_number):
                if (iterations + 1) % config.display == 0:
                    write_loss(iterations, max_iterations, trainer, train_writer)
                if (iterations + 1) % config.image_save_iterations == 0:
                    img_filename = '%s/gen_%08d.jpg' % (image_directory, iterations + 1)
                    torchvision.utils.save_image(assembled_images.data / 2 + 0.5, img_filename, nrow = 1)
                    write_html(snapshot_directory + '/index.html', iterations + 1, config.image_save_iterations, image_directory)
                elif (iterations + 1) % config.image_display_iterations == 0:
                    img_filename = '%s/gen.jpg' % (image_directory)
                    torchvision.utils.save_image(assembled_images.data / 2 + 0.5, img_filename, nrow = 1)

                if (iterations + 1) % config.snapshot_save_iterations == 0:
                    trainer.save(config.snapshot_prefix, iterations)

                iterations += 1
                if iterations >= max_iterations:
                    return

if __name__ == '__main__':
    main(sys.argv)
