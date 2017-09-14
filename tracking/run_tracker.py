#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Set binary and encoding so that I can use fancy unicode symbols like ✓ in my code

# external library & framework import
import numpy as np
import os
import sys
import time
import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt

# Torch imports
import torch
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

# Using skvideo.io for reading video files, because opencv2's reader somehow doesn't work on my system (so far, probably going to change that at some point)
# skvideo.io.FFmpegReader does a good job, but seems to shift the colors. If you want cv2.Capture (or however it's called), just fork it and make a pull request
# but please... enable us to choose between the two
import skvideo.io
# Using untangle to read xml
import untangle
# Using call to get our modified darknet output
import subprocess #using subprocess.check_output
# Using OpenCV to save images
import cv2

# Local file imports
sys.path.insert(0,'../modules')
from sample_generator import *
from data_prov import *
from model import *
from bbreg import *
from options import *
from gen_config import *

np.random.seed(123)
torch.manual_seed(456)
#torch.cuda.manual_seed(789)

opts['use_gpu'] = False

# My variables, let's make this thing more ... adaptable in look and fell as well as algorythm behavior


# STARTING POINT VARIABLES
# Using the starting point to make a rectangle out of the first bbox
global START_POINT_HEIGHT=3 # Height of the starting point rectangle
global START_POINT_WIDTH=3 # Width of the starting point rectangle

# EMERGENCY MODE VARIABLES
# Using the emergency mode to reduce the threshold so that we are able to track the object through "full" occlusions (snapping back onto it after it leaves the occlusion)
# This paramater may need some fine-tuning depending on the video
global EMERGENCY_MODE = False
global EMERGENCY_MODE_THRESHOLD = -2
global EMERGENCY_MODE_WAIT_FRAMES=50

# FILE VARIABLES
global VIDEO_SRC = "../trafficvid1.mp4"
global YOLO_OUTPUT_DIR = "../yolo_output"
global ORIGINAL_FRAME_JPG_NAME = "OG_FRAME.jpg"

# Classes that should be tracked
global TRACK_CLASSES = ["car", "bus", "truck", "person"] #we got a traffic video so, let's track traffic

# TODO
# 1. Try to track 2 objects at a time
# 2. Use darknet to get the coordinates of detected objects
# 3. Track the detected objects if their overlap threshold with tracker.coords is < 0.9 (n)
# 4. Try livetracking a current crossover

# TODO
# - Use Region of DISinterest in order to kill the tracker if the object has reached one of the destinies
# - Emergency mode in which the threshold gets lowered to a reasonable point in order to get back on track (eg. after occlusion)
# - Implement an RNN, input: coordinates | output: estimation of where the car should be, can be used for estimation when there's occlusion or for additional safety measure when showing it normally

# TODO next
# 1. implement emergency mode - ✓
# 2. find a way to use a pretrained model, so that it doesn't retrain at every start - ✓ (i'm ... i've not acted smart (;)) and didn't understand what "train" actually meant ..)
# 3. Create a tracker class, in which variables are set after each tick, you have getters/setters, easily maintainable, ...


def forward_samples(model, image, samples, out_layer='conv3'):
    model.eval()
    extractor = RegionExtractor(image, samples, opts['img_size'], opts['padding'], opts['batch_test'])
    for i, regions in enumerate(extractor):
        regions = Variable(regions)
        if opts['use_gpu']:
            regions = regions.cuda()
        feat = model(regions, out_layer=out_layer)
        if i==0:
            feats = feat.data.clone() #somehow this f*cks up if there are certian groundtruth_rect.txt coordinates given (mine didn't work, TODO investigate this later)
        else:
            feats = torch.cat((feats,feat.data.clone()),0)
    return feats


def set_optimizer(model, lr_base, lr_mult=opts['lr_mult'], momentum=opts['momentum'], w_decay=opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.iteritems():
        lr = lr_base
        for l, m in lr_mult.iteritems():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr':lr})
    optimizer = optim.SGD(param_list, lr = lr, momentum=momentum, weight_decay=w_decay)
    return optimizer


def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
    model.train()

    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while(len(pos_idx) < batch_pos*maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while(len(neg_idx) < batch_neg_cand*maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for iter in range(maxiter):

        # select pos idx
        pos_next = pos_pointer+batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer+batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = Variable(pos_feats.index_select(0, pos_cur_idx))
        batch_neg_feats = Variable(neg_feats.index_select(0, neg_cur_idx))

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0,batch_neg_cand,batch_test):
                end = min(start+batch_test,batch_neg_cand)
                score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start==0:
                    neg_cand_score = score.data[:,1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.data[:,1].clone()),0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats.index_select(0, Variable(top_idx))
            model.train()

        # forward
        pos_score = model(batch_pos_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)

        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
        optimizer.step()

        #print "Iter %d, Loss %.4f" % (iter, loss.data[0])





# "Tracker"-Function, this trains the
def run_mdnet(img_list, init_bbox, gt=None, savefig_dir='', display=False):

    # Init bbox
    target_bbox = np.array(init_bbox)
    result = np.zeros((len(img_list),4))
    result_bb = np.zeros((len(img_list),4))
    result[0] = target_bbox
    result_bb[0] = target_bbox

    # Init model
    model = MDNet(opts['model_path'])
    if opts['use_gpu']:
        model = model.cuda()
    else:
        print("Not using CUDA")
    model.set_learnable_params(opts['ft_layers'])


    # Init criterion and optimizer
    criterion = BinaryLoss()
    init_optimizer = set_optimizer(model, opts['lr_init'])
    update_optimizer = set_optimizer(model, opts['lr_update'])

    tic = time.time()
    # Load first image
    image = Image.open(img_list[0]).convert('RGB')
    print("opened image")

    # Train bbox regressor
    bbreg_examples = gen_samples(SampleGenerator('uniform', image.size, 0.3, 1.5, 1.1),
                                 target_bbox, opts['n_bbreg'], opts['overlap_bbreg'], opts['scale_bbreg'])
    bbreg_feats = forward_samples(model, image, bbreg_examples)
    bbreg = BBRegressor(image.size)
    bbreg.train(bbreg_feats, bbreg_examples, target_bbox)

    # Getting positive examples
    pos_examples = gen_samples(SampleGenerator('gaussian', image.size, 0.1, 1.2),
                               target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

    # Getting negative image examples
    neg_examples = np.concatenate([
                    gen_samples(SampleGenerator('uniform', image.size, 1, 2, 1.1),
                                target_bbox, opts['n_neg_init']//2, opts['overlap_neg_init']),
                    gen_samples(SampleGenerator('whole', image.size, 0, 1.2, 1.1),
                                target_bbox, opts['n_neg_init']//2, opts['overlap_neg_init'])])
    #
    neg_examples = np.random.permutation(neg_examples)

    # Extract pos/neg features
    pos_feats = forward_samples(model, image, pos_examples)
    neg_feats = forward_samples(model, image, neg_examples)
    feat_dim = pos_feats.size(-1)

    # pos_feats/neg_feats contain the features that the convnet should look out for!
    print("Training started")
    # Initial training
    train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init']) #The model gets trained to watch those features
    print("Training finished")

    # Init sample generators
    sample_generator = SampleGenerator('gaussian', image.size, opts['trans_f'], opts['scale_f'], valid=True)
    pos_generator = SampleGenerator('gaussian', image.size, 0.1, 1.2)
    neg_generator = SampleGenerator('uniform', image.size, 1.5, 1.2)

    # Init pos/neg features for update
    pos_feats_all = [pos_feats[:opts['n_pos_update']]]
    neg_feats_all = [neg_feats[:opts['n_neg_update']]]

    spf_total = time.time()-tic

    # Display
    savefig = savefig_dir != ''
    if display or savefig:
        dpi = 80.0
        figsize = (image.size[0]/dpi, image.size[1]/dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image, aspect='normal')

        if gt is not None:
            gt_rect = plt.Rectangle(tuple(gt[0,:2]),gt[0,2],gt[0,3],
                    linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(gt_rect)

        rect = plt.Rectangle(tuple(result_bb[0,:2]),result_bb[0,2],result_bb[0,3],
                linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        ax.add_patch(rect)

        if display:
            plt.pause(.01)
            plt.draw()
        if savefig:
            fig.savefig(os.path.join(savefig_dir,'0000.jpg'),dpi=dpi)




    # This is where the actual tracking happens
    # Main loop
    print("Starting main loop")
    EMERGENCY_MODE_FRAMES_COUNTER = 0
    for i in range(1,len(img_list)):

        tic = time.time()
        # Load image
        image = Image.open(img_list[i]).convert('RGB')


        # Estimate target bbox
        samples = gen_samples(sample_generator, target_bbox, opts['n_samples'])
        sample_scores = forward_samples(model, image, samples, out_layer='fc6')
        top_scores, top_idx = sample_scores[:,1].topk(5)
        top_idx = top_idx.cpu().numpy()
        target_score = top_scores.mean()
        target_bbox = samples[top_idx].mean(axis=0)

        global success
        # Enabling / Disabling EMERGENCY_MODE
        if target_score < opts['success_thr']:
            # Seems like we lost the object, set the emergency threshold and wait for the defined amount of frames
            EMERGENCY_MODE = True
            print("Enabled emergency mode, lowering threshold to: " + str(EMERGENCY_MODE_THRESHOLD))
            if EMERGENCY_MODE_FRAMES_COUNTER == EMERGENCY_MODE_WAIT_FRAMES:
                #the frame counter reached the end
                # TODO kill this tracker
                exit()
            EMERGENCY_MODE_FRAMES_COUNTER += 1 #increasing the frame counter

            # setting success variable so that we actually run the code we want to run in emergency mode
            success = target_score > EMERGENCY_MODE_THRESHOLD
        else:
            if EMERGENCY_MODE:
                print("Disabled emergency mode, setting threshold back to: " + str(opts['success_thr'])) #printing disable message
            EMERGENCY_MODE = False #disabling emergency mode
            # Doing the normal threshold comparison since this is the non-emergency mode
            success = target_score > opts['success_thr']


        print("Success:", success)
        # Expand search area at failure
        if success:
            sample_generator.set_trans_f(opts['trans_f']) #Everything fine, use normal search area
        else:
            sample_generator.set_trans_f(opts['trans_f_expand']) #If success=False, expand the search

        # REMARK: Area expansion and the emergency mode are (tested with one vid so far) able to get back on track with a fully occluded object

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            bbreg_feats = forward_samples(model, image, bbreg_samples)
            bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox

        # Copy previous result at failure
        if not success:
            target_bbox = result[i-1]
            bbreg_bbox = result_bb[i-1]

        # Save result
        result[i] = target_bbox
        result_bb[i] = bbreg_bbox

        # Data collect
        print("Collection data")
        if success:
            # Draw pos/neg samples
            pos_examples = gen_samples(pos_generator, target_bbox,
                                       opts['n_pos_update'],
                                       opts['overlap_pos_update'])
            neg_examples = gen_samples(neg_generator, target_bbox,
                                       opts['n_neg_update'],
                                       opts['overlap_neg_update'])

            # Extract pos/neg features
            pos_feats = forward_samples(model, image, pos_examples)
            neg_feats = forward_samples(model, image, neg_examples)
            pos_feats_all.append(pos_feats)
            neg_feats_all.append(neg_feats)
            if len(pos_feats_all) > opts['n_frames_long']:
                del pos_feats_all[0]
            if len(neg_feats_all) > opts['n_frames_short']:
                del neg_feats_all[0]

        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'],len(pos_feats_all))
            pos_data = torch.stack(pos_feats_all[-nframes:],0).view(-1,feat_dim)
            neg_data = torch.stack(neg_feats_all,0).view(-1,feat_dim)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        # Long term update
        elif i % opts['long_interval'] == 0:
            pos_data = torch.stack(pos_feats_all,0).view(-1,feat_dim)
            neg_data = torch.stack(neg_feats_all,0).view(-1,feat_dim)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])



        spf = time.time()-tic
        spf_total += spf

        # Display
        if display or savefig:
            im.set_data(image)

            start_point_center = {}
            start_point_center["x"] = ((gt[i,2]/2)+gt[i,0]) #x=x+(w/2)
            start_point_center["y"] = ((gt[i,3]/2)+gt[i,1]) #y=y+(h/2)
            start_point_center["h"] = START_POINT_HEIGHT
            start_point_center["w"] = START_POINT_WIDTH


            # Setting the groundtruth_rect (start point, for me though.)
            gt_rect.set_xy([start_point_center["x"], start_point_center["y"]]) #setting x and y in array/list
            gt_rect.set_width(start_point_center["w"])
            gt_rect.set_height(start_point_center["h"])

            # Setting the tracked rectangle
            rect.set_xy(result_bb[i,:2])
            rect.set_width(result_bb[i,2])
            rect.set_height(result_bb[i,3])


            # @TODO
            # use these coordinates to check the overlap

            if display:
                plt.pause(.01)
                plt.draw()
            if savefig:
                fig.savefig(os.path.join(savefig_dir,'%04d.jpg'%(i)),dpi=dpi)
                print("Saved figure ("+str(i).zfill(4)+".jpg).")

            # Not printing overlap of gt_rect and rect because gt_rect serves the sole purpose of setting our starting point
            print "Frame %d/%d, Score %.3f, Time %.3f" % \
            (i, len(img_list), target_score, spf)
    fps = len(img_list) / spf_total
    return result, result_bb, fps

# Creates a tracker and does some checks in before
def createTracker(x1, y1, x2, y2, trackers):
    # Check if:
    # 1. x&y are in image bounds
    # 2. object is already being tracked
    # 3.

# Using this function to handle everything (managing nets, feeding frames, ...)
def controlNets():
    frame_counter = 0
    reader = skvideo.io.FFmpegReader(VIDEO_SRC) #initialize the reader

    for frame in reader.nextFrame(): #reading frame by frame
        # 1. Use YOLO to get the cars in this frame
        if frame_counter == 0:
            cv2.imsave(YOLO_OUTPUT_DIR + ORIGINAL_FRAME_JPG_NAME) #saving image
            # Since this is frame 0, simply let YOLO run through the whole image
            output = subprocess.check_output("./darknet detector test cfg/coco.data cfg/yolo.cfg yolo.weights " + ORIGINAL_FRAME_JPG_NAME, shell=True) # Calling darknet and saving it's output
            # Get the coordinates of the detected objects, filter them through name and ROI coordinates
            for line in output.splitlines():
                if line[:1] == "<": #check if the first character is a "<" -> that way we know it's our output and parsable xml
                    obj = untangle.parse(line)
                    if obj.detected.name.cdata in TRACK_CLASSES: #if name of detected object is in list (eg. car), we can go on and track it
                        # start to track the object, here the real stuff comes in


        frame_counter += 1


    # 1. Get necessary data
    # 2. Start the network(s) / tracker(s)
    # 3. Rewrite the networks so that the location data gets saved automatically (or return it here)
    # 4. If a network runs out of something to do, kill it (this will most likely handle itself)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', default='', help='input seq')
    parser.add_argument('-j', '--json', default='', help='input json')
    parser.add_argument('-f', '--savefig', action='store_true')
    parser.add_argument('-d', '--display', action='store_true')

    args = parser.parse_args()

    # Replaced assertion with simple if-block... simply looks better :)
    if not (args.seq != '' or args.json != ''):
        print("[EXIT] Please pass proper & enough parameters!")
        exit()

    # Generate sequence config -> basically just some configuration parameters
    img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args)

    # Run Tracker
    # TODO check if it is possible to multiply this tracker
    result, result_bb, fps = run_mdnet(img_list, init_bbox, gt=gt, savefig_dir=savefig_dir, display=display)



    # Save result
    res = {}
    res['res'] = result_bb.round().tolist()
    res['type'] = 'rect'
    res['fps'] = fps
    json.dump(res, open(result_path, 'w'), indent=2)
