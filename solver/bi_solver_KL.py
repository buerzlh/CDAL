import torch
import torch.nn as nn
import os
from . import utils as solver_utils
from utils.utils import to_cuda, to_onehot
from torch import optim
from . import clustering
from discrepancy.cdd import CDD
from math import ceil as ceil
from .base_solver import BaseSolver
from copy import deepcopy
from datetime import datetime 

class BiSolver(BaseSolver):
    def __init__(self, net, dataloader, bn_domain_map={}, resume=None, **kwargs):
        super(BiSolver, self).__init__(net, dataloader, \
                      bn_domain_map=bn_domain_map, resume=resume, **kwargs)

        if len(self.bn_domain_map) == 0:
            self.bn_domain_map = {self.source_name: 0, self.target_name: 1}

        self.clustering_source_name = 'clustering_' + self.source_name
        self.clustering_target_name = 'clustering_' + self.target_name

        assert('categorical' in self.train_data)


        num_layers = len(self.net.module.FC1) + 2
        self.cdd = CDD(kernel_num=self.opt.CDD.KERNEL_NUM, kernel_mul=self.opt.CDD.KERNEL_MUL,
                  num_layers=num_layers, num_classes=self.opt.DATASET.NUM_CLASSES, 
                  intra_only=self.opt.CDD.INTRA_ONLY)

        self.discrepancy_key = 'intra' if self.opt.CDD.INTRA_ONLY else 'cdd'
        self.clustering = clustering.Clustering(self.opt.CLUSTERING.EPS, 
                                        self.opt.CLUSTERING.FEAT_KEY, 
                                        self.opt.CLUSTERING.BUDGET)

        self.clustered_target_samples = {}

    def complete_training(self):
        if self.loop >= self.opt.TRAIN.MAX_LOOP:
            return True

        if 'target_centers' not in self.history or \
                'ts_center_dist' not in self.history or \
                'target_labels' not in self.history:
            return False

        if len(self.history['target_centers']) < 2 or \
		len(self.history['ts_center_dist']) < 1 or \
		len(self.history['target_labels']) < 2:
           return False

        # target centers along training
        target_centers = self.history['target_centers']
        eval1 = torch.mean(self.clustering.Dist.get_dist(target_centers[-1], 
			target_centers[-2])).item()

        # target-source center distances along training
        eval2 = self.history['ts_center_dist'][-1].item()

        # target labels along training
        path2label_hist = self.history['target_labels']
        paths = self.clustered_target_samples['data']
        num = 0
        for path in paths:
            pre_label = path2label_hist[-2][path]
            cur_label = path2label_hist[-1][path]
            if pre_label != cur_label:
                num += 1
        eval3 = 1.0 * num / len(paths)

        return (eval1 < self.opt.TRAIN.STOP_THRESHOLDS[0] and \
                eval2 < self.opt.TRAIN.STOP_THRESHOLDS[1] and \
                eval3 < self.opt.TRAIN.STOP_THRESHOLDS[2])

    def solve(self):
        stop = False
        self.max_acc = 0.0
        if self.resume:
            self.iters += 1
            self.loop += 1

        while True: 
            # updating the target label hypothesis through clustering
            target_hypt = {}
            filtered_classes = []
            with torch.no_grad():
                #self.update_ss_alignment_loss_weight()
                print('Clustering based on %s...' % self.source_name)
                self.update_labels()
                self.clustered_target_samples = self.clustering.samples
                target_centers = self.clustering.centers 
                center_change = self.clustering.center_change 
                path2label = self.clustering.path2label

                # updating the history
                self.register_history('target_centers', target_centers,
	            	self.opt.CLUSTERING.HISTORY_LEN)
                self.register_history('ts_center_dist', center_change,
	            	self.opt.CLUSTERING.HISTORY_LEN)
                self.register_history('target_labels', path2label,
	            	self.opt.CLUSTERING.HISTORY_LEN)

                if self.clustered_target_samples is not None and \
                              self.clustered_target_samples['gt'] is not None:
                    preds = to_onehot(self.clustered_target_samples['label'], 
                                                self.opt.DATASET.NUM_CLASSES)
                    gts = self.clustered_target_samples['gt']
                    res = self.model_eval(preds, gts)
                    print('Clustering %s: %.4f' % (self.opt.EVAL_METRIC, res))

                # check if meet the stop condition
                stop = self.complete_training()
                if stop: break
                
                # filtering the clustering results
                target_hypt, filtered_classes = self.filtering()

                # update dataloaders
                self.construct_categorical_dataloader(target_hypt, filtered_classes)
                # update train data setting
                self.compute_iters_per_loop(filtered_classes)

            # k-step update of network parameters through forward-backward process
            self.update_network(filtered_classes)
            self.loop += 1
        save_path = self.opt.SAVE_DIR
        acc = str(round(self.max_acc,2))
        out = save_path.split('/')
        out.pop()
        newout = ''
        for m in out:
            newout = os.path.join(newout,m)
        # import datetime
        nowTime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        newout = os.path.join(newout,acc+'_'+str(nowTime))
        os.rename(save_path, newout)
        print('Training Done!')
        
    def update_labels(self):
        net = self.net
        net.eval()
        opt = self.opt

        source_dataloader = self.train_data[self.clustering_source_name]['loader']
        net.module.set_bn_domain(self.bn_domain_map[self.source_name])

        source_centers = solver_utils.get_centers(net, 
		source_dataloader, self.opt.DATASET.NUM_CLASSES, 
                self.opt.CLUSTERING.FEAT_KEY)
        init_target_centers = source_centers

        target_dataloader = self.train_data[self.clustering_target_name]['loader']
        net.module.set_bn_domain(self.bn_domain_map[self.target_name])

        self.clustering.set_init_centers(init_target_centers)
        self.clustering.feature_clustering(net, target_dataloader)
        self.path2label = self.clustering.path2label
        

    def filtering(self):
        threshold = self.opt.CLUSTERING.FILTERING_THRESHOLD
        min_sn_cls = self.opt.TRAIN.MIN_SN_PER_CLASS
        target_samples = self.clustered_target_samples

        # filtering the samples
        chosen_samples = solver_utils.filter_samples(
		target_samples, threshold=threshold)

        # filtering the classes
        filtered_classes = solver_utils.filter_class(
		chosen_samples['label'], min_sn_cls, self.opt.DATASET.NUM_CLASSES)

        print('The number of filtered classes: %d.' % len(filtered_classes))
        return chosen_samples, filtered_classes

    def construct_categorical_dataloader(self, samples, filtered_classes):
        # update self.dataloader
        target_classwise = solver_utils.split_samples_classwise(
			samples, self.opt.DATASET.NUM_CLASSES)

        dataloader = self.train_data['categorical']['loader']
        classnames = dataloader.classnames
        dataloader.class_set = [classnames[c] for c in filtered_classes]
        dataloader.target_paths = {classnames[c]: target_classwise[c]['data'] \
                      for c in filtered_classes}
        dataloader.num_selected_classes = min(self.opt.TRAIN.NUM_SELECTED_CLASSES, len(filtered_classes))
        dataloader.construct()

    def CAS(self):
        samples = self.get_samples('categorical')

        source_samples = samples['Img_source']
        source_sample_paths = samples['Path_source']
        source_nums = [len(paths) for paths in source_sample_paths]

        target_samples = samples['Img_target']
        target_sample_paths = samples['Path_target']
        target_nums = [len(paths) for paths in target_sample_paths]
        
        source_sample_labels = samples['Label_source']
        self.selected_classes = [labels[0].item() for labels in source_sample_labels]
        assert(self.selected_classes == 
               [labels[0].item() for labels in  samples['Label_target']])
        return source_samples, source_nums, target_samples, target_nums
            
    def prepare_feats(self, feats):
        return [feats[key] for key in feats if key in self.opt.CDD.ALIGNMENT_FEAT_KEYS]

    def compute_iters_per_loop(self, filtered_classes):
        self.iters_per_loop = int(len(self.train_data['categorical']['loader'])) * self.opt.TRAIN.UPDATE_EPOCH_PERCENTAGE
        print('Iterations in one loop: %d' % (self.iters_per_loop))

    def update_network(self, filtered_classes):
        # initial configuration
        stop = False
        update_iters = 0

        self.train_data[self.source_name]['iterator'] = \
                     iter(self.train_data[self.source_name]['loader'])
        self.train_data[self.target_name]['iterator'] = \
                     iter(self.train_data[self.target_name]['loader'])
        self.train_data['categorical']['iterator'] = \
                     iter(self.train_data['categorical']['loader'])

        while not stop:
            # update learning rate
            self.update_lr()

            # set the status of network
            self.net.train()

            # coventional sampling for training on labeled source data
            source_sample = self.get_samples(self.source_name)
            target_sample = self.get_samples(self.target_name) 
            source_data, source_gt = source_sample['Img'],\
                          source_sample['Label']
            target_data = target_sample['Img']
            target_path = target_sample['Path']
            target_label = self.get_label(target_path)

            source_data = to_cuda(source_data)
            source_gt = to_cuda(source_gt)
            self.net.module.set_bn_domain(self.bn_domain_map[self.source_name])
            source_result = self.net(source_data)
            source_preds1 = source_result['logits1']
            source_preds2 = source_result['logits2']

            # compute the cross-entropy loss
            loss_A = self.CELoss(source_preds1, source_gt) + self.CELoss(source_preds2, source_gt)
            self.optimizer['G'].zero_grad()
            self.optimizer['FC1'].zero_grad()
            self.optimizer['FC2'].zero_grad()
            loss_A.backward()
            self.optimizer['G'].step()
            self.optimizer['FC1'].step()
            self.optimizer['FC2'].step()


            self.net.module.set_bn_domain(self.bn_domain_map[self.source_name])
            source_result = self.net(source_data)
            source_preds1 = source_result['logits1']
            source_preds2 = source_result['logits2']

            # compute the cross-entropy loss
            loss_B = self.CELoss(source_preds1, source_gt) + self.CELoss(source_preds2, source_gt)

            self.net.module.set_bn_domain(self.bn_domain_map[self.target_name])
            target_result = self.net(target_data)

            pred_u1 = target_result['probs1']
            pred_u2 = target_result['probs2']
            with torch.no_grad():
                target_1 = target_result['logits1']
                target_2 = target_result['logits2']
                loss_1 = nn.CrossEntropyLoss(reduction='none')(target_1, target_label)  ## KL is equal to the cross-entropy when using pseudo label
                loss_2 = nn.CrossEntropyLoss(reduction='none')(target_2, target_label) 

                mask_1 = loss_1 < loss_2
                mask_2 = loss_1 > loss_2
                
            pred_u1_1 = pred_u1[mask_1]
            pred_u2_1 = pred_u2[mask_1]

            pred_u1_2 = pred_u1[mask_2]
            pred_u2_2 = pred_u2[mask_2]
            loss_dis_1 = -self.discrepancy(pred_u1_1, pred_u2_1.detach()) + self.discrepancy(pred_u1_2, pred_u2_2.detach())
            loss_dis_2 = self.discrepancy(pred_u1_1.detach(), pred_u2_1) - self.discrepancy(pred_u1_2.detach(), pred_u2_2)

            loss_B = loss_B + loss_dis_1 + loss_dis_2

            self.optimizer['FC1'].zero_grad()
            self.optimizer['FC2'].zero_grad()
            loss_B.backward()
            self.optimizer['FC1'].step()
            self.optimizer['FC2'].step()

            for _ in range(4):
                target_result = self.net(target_data)
                pred_u1 = target_result['probs1']
                pred_u2 = target_result['probs2']
                loss_step_C = self.discrepancy(pred_u1, pred_u2)
                self.optimizer['G'].zero_grad()
                loss_step_C.backward()
                self.optimizer['G'].step()


            if len(filtered_classes) > 0:
                source_samples_cls, source_nums_cls, \
                       target_samples_cls, target_nums_cls = self.CAS()    

                source_cls_concat = torch.cat([to_cuda(samples) 
                            for samples in source_samples_cls], dim=0)
                target_cls_concat = torch.cat([to_cuda(samples) 
                            for samples in target_samples_cls], dim=0)

                self.net.module.set_bn_domain(self.bn_domain_map[self.source_name])
                feats_source = self.net(source_cls_concat)
                self.net.module.set_bn_domain(self.bn_domain_map[self.target_name])
                feats_target = self.net(target_cls_concat)

                feats_toalign_S = self.prepare_feats(feats_source)
                feats_toalign_T = self.prepare_feats(feats_target)                 

                loss_C = self.cdd.forward(feats_toalign_S, feats_toalign_T, 
                               source_nums_cls, target_nums_cls)[self.discrepancy_key]

                loss_C *= self.opt.CDD.LOSS_WEIGHT
                self.optimizer['G'].zero_grad()
                loss_C.backward()
                self.optimizer['G'].step()
                loss_step_C = loss_step_C + loss_C

            

            if self.opt.TRAIN.LOGGING and (update_iters+1) % \
                      (max(1, self.iters_per_loop // self.opt.TRAIN.NUM_LOGGING_PER_LOOP)) == 0:
                accu = self.model_eval((source_preds1+source_preds2)/2, source_gt)
                cur_loss = {'loss_A': loss_A, 'loss_B': loss_B,
			'loss_C': loss_step_C}
                self.logging(cur_loss, accu)

            self.opt.TRAIN.TEST_INTERVAL = min(1.0, self.opt.TRAIN.TEST_INTERVAL)
            self.opt.TRAIN.SAVE_CKPT_INTERVAL = min(1.0, self.opt.TRAIN.SAVE_CKPT_INTERVAL)

            if self.opt.TRAIN.TEST_INTERVAL > 0 and \
		(update_iters+1) % int(self.opt.TRAIN.TEST_INTERVAL * self.iters_per_loop) == 0:
                with torch.no_grad():
                    self.net.module.set_bn_domain(self.bn_domain_map[self.target_name])
                    self.temp_accu = self.test()
                    print('Test at (loop %d, iters: %d) with %s: %.4f.' % (self.loop, 
                              self.iters, self.opt.EVAL_METRIC, self.temp_accu))
                    print('max acc:' + str(self.max_acc))

            if self.opt.TRAIN.SAVE_CKPT_INTERVAL > 0 and \
        (update_iters+1) % int(self.opt.TRAIN.SAVE_CKPT_INTERVAL * self.iters_per_loop) == 0:
                if self.temp_accu > self.max_acc:
                    self.max_acc = self.temp_accu
                    self.save_ckpt()


            update_iters += 1
            self.iters += 1

            # update stop condition
            if update_iters >= self.iters_per_loop:
                stop = True
            else:
                stop = False


    def get_label(self, path):
        label = []
        l = len(path)
        for i in range(l):
            label.append(self.path2label[path[i]])
        label  = to_cuda(torch.Tensor(label).long())
        return label

    def discrepancy(self, p1, p2):
        # s = p1.shape
        # if s[1] > 1:
        #     proj = torch.randn(s[1], 128).to(self.device)
        #     proj *= torch.rsqrt(torch.sum(torch.mul(proj, proj), 0, keepdim=True))
        #     p1 = torch.matmul(p1, proj)
        #     p2 = torch.matmul(p2, proj)
        # p1 = torch.topk(p1, s[0], dim=0)[0]
        # p2 = torch.topk(p2, s[0], dim=0)[0]
        # dist = p1 - p2
        # wdist = torch.mean(torch.mul(dist, dist))
        # return wdist
        return (p1 - p2).abs().mean()

    def save_ckpt(self):
        save_path = self.opt.SAVE_DIR
        ckpt_resume = os.path.join(save_path, 'ckpt_max.resume')
        ckpt_weights = os.path.join(save_path, 'ckpt_max.weights')
        torch.save({'loop': self.loop,
                    'iters': self.iters,
                    'model_state_dict': self.net.module.state_dict(),
                    # 'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.history,
                    'bn_domain_map': self.bn_domain_map
                    }, ckpt_resume)

        torch.save({'weights': self.net.module.state_dict(),
                    'bn_domain_map': self.bn_domain_map
                    }, ckpt_weights)