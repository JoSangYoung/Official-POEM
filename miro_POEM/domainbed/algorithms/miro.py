# Copyright (c) Kakao Brain. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from domainbed.optimizers import get_optimizer
from domainbed.networks.ur_networks import URFeaturizer
from domainbed.lib import misc
from domainbed.algorithms import Algorithm


class ForwardModel(nn.Module):
    """Forward model is used to reduce gpu memory usage of SWAD.
    """
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        return self.predict(x)

    def predict(self, x):
        return self.network(x)

    def predict_domain(self, x):
        return self.network(x)


class MeanEncoder(nn.Module):
    """Identity function"""
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x


class VarianceEncoder(nn.Module):
    """Bias-only model with diagonal covariance"""
    def __init__(self, shape, init=0.1, channelwise=True, eps=1e-5):
        super().__init__()
        self.shape = shape
        self.eps = eps

        init = (torch.as_tensor(init - eps).exp() - 1.0).log()
        b_shape = shape
        if channelwise:
            if len(shape) == 4:
                # [B, C, H, W]
                b_shape = (1, shape[1], 1, 1)
            elif len(shape ) == 3:
                # CLIP-ViT: [H*W+1, B, C]
                b_shape = (1, 1, shape[2])
            else:
                raise ValueError()

        self.b = nn.Parameter(torch.full(b_shape, init))

    def forward(self, x):
        return F.softplus(self.b) + self.eps


def get_shapes(model, input_shape):
    # get shape of intermediate features
    with torch.no_grad():
        dummy = torch.rand(1, *input_shape).to(next(model.parameters()).device)
        _, feats = model(dummy, ret_feats=True)
        shapes = [f.shape for f in feats]

    return shapes


class MIRO(Algorithm):
    """Mutual-Information Regularization with Oracle"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, **kwargs):
        super().__init__(input_shape, num_classes, num_domains, hparams)

        self.bool_angle = hparams.bool_angle
        self.bool_task = hparams.bool_task

        self.pre_featurizer = URFeaturizer(
            input_shape, self.hparams, freeze="all", feat_layers=hparams.feat_layers
        )
        self.featurizer = URFeaturizer(
            input_shape, self.hparams, feat_layers=hparams.feat_layers
        )
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.ld = hparams.ld

        # build mean/var encoders
        shapes = get_shapes(self.pre_featurizer, self.input_shape)
        self.mean_encoders = nn.ModuleList([
            MeanEncoder(shape) for shape in shapes
        ])
        self.var_encoders = nn.ModuleList([
            VarianceEncoder(shape) for shape in shapes
        ])

        # optimizer
        parameters = [
            {"params": self.network.parameters()},
            {"params": self.mean_encoders.parameters(), "lr": hparams.lr * hparams.lr_mult},
            {"params": self.var_encoders.parameters(), "lr": hparams.lr * hparams.lr_mult},
        ]
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )


        if self.bool_angle or self.bool_task:
            self.domain_hparam = self.hparams
            self.domain_hparam["domain"] = True

            self.featurizer_domain = URFeaturizer(
                input_shape, self.hparams, feat_layers=hparams.feat_layers
            )
            self.classifier_domain = nn.Linear(self.featurizer_domain.n_outputs, num_classes)
            self.network_domain = nn.Sequential(self.featurizer_domain, self.classifier_domain)
            # self.ld_domain = hparams.ld_domain

            # self.mean_encoders_domain = nn.ModuleList([
            #     MeanEncoder(shape) for shape in shapes
            # ])
            # self.var_encoders_domain = nn.ModuleList([
            #     VarianceEncoder(shape) for shape in shapes
            # ])

            # parameters_domain = [
            #     {"params": self.network_domain.parameters()},
            #     {"params": self.mean_encoders_domain.parameters(), "lr": hparams.lr * hparams.lr_mult},
            #     {"params": self.var_encoders_domain.parameters(), "lr": hparams.lr * hparams.lr_mult},
            # ]
            self.optimizer_domain = get_optimizer(
                hparams["optimizer"],
                self.network_domain.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        if self.bool_task:
            self.classifier_task = nn.Linear(self.featurizer_domain.n_outputs, 2)
            self.optimizer_task = get_optimizer(
                hparams["optimizer"],
                self.classifier_task.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

    def update(self, x, y, d, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        feature_class, inter_feats = self.featurizer(all_x, ret_feats=True)
        logit = self.classifier(feature_class)
        loss = F.cross_entropy(logit, all_y)

        # MIRO
        with torch.no_grad():
            _, pre_feats = self.pre_featurizer(all_x, ret_feats=True)

        reg_loss = 0.
        for f, pre_f, mean_enc, var_enc in misc.zip_strict(
            inter_feats, pre_feats, self.mean_encoders, self.var_encoders
        ):
            # mutual information regularization
            mean = mean_enc(f)
            var = var_enc(f)
            vlb = (mean - pre_f).pow(2).div(var) + var.log()
            reg_loss += vlb.mean() / 2.

        loss += reg_loss * self.ld

        output = {"loss":loss.item()}

        self.optimizer.zero_grad()

        if self.bool_angle or self.bool_task:
            network_list = {"class_feature": self.featurizer, "domain_feature": self.featurizer_domain,
                            "class_classifier": self.classifier, "domain_classifier": self.classifier_domain}
            if self.bool_task:
                network_list["task_classifier"] = self.classifier_task

            all_d = torch.cat(d)
            feature_domain = self.featurizer_domain(all_x)

            self.optimizer_domain.zero_grad()

            # # MIRO
            # with torch.no_grad():
            #     _, pre_feats = self.pre_featurizer(all_x, ret_feats=True)

            # reg_loss_domain = 0.
            # for f, pre_f, mean_enc, var_enc in misc.zip_strict(
            #     inter_feats, pre_feats, self.mean_encoders_domain, self.var_encoders_domain
            # ):
            #     # mutual information regularization
            #     mean = mean_enc(f)
            #     var = var_enc(f)
            #     vlb = (mean - pre_f).pow(2).div(var) + var.log()
            #     reg_loss_domain += vlb.mean() / 2.

            loss_domain = F.cross_entropy(self.predict_domain(all_x), all_d)
            # loss_domain += reg_loss_domain * self.ld_domain
            
            output["loss_domain"] = loss_domain.item()



            if self.bool_angle:
                loss_angle = torch.abs(F.cosine_similarity(feature_class, feature_domain, dim=1))
                loss_angle = torch.mean(loss_angle)
                output["angle_loss"] = loss_angle.item()

            if self.bool_task:
                task_label = [0] * all_d.shape[0] + [1] * all_y.shape[0]
                task_label = torch.tensor(task_label).to("cuda")
                task_features = torch.tensor(torch.cat((feature_class.clone(), feature_domain.clone()))).to("cuda")
                loss_task = F.cross_entropy(self.classifier_task(task_features), task_label)
                output["task_loss"] = loss_task.item()


            for key in network_list.keys():
                if "class_classifier" in key or "domain_classifier" in key:
                    for param in network_list[key].parameters():
                        param.requires_grad = True
                else:
                    for param in network_list[key].parameters():
                        param.requires_grad = False

            loss.backward(retain_graph=True)
            loss_domain.backward(retain_graph=True)

            if self.bool_angle:
                loss = loss + loss_angle
                loss_domain = loss_domain + loss_angle

            if self.bool_task:
                loss = loss + loss_task
                loss_domain = loss_domain + loss_task

        if self.bool_angle or self.bool_task:
            for key in network_list.keys():
                if "domain_feature" in key or "task" in key:
                    for param in network_list[key].parameters():
                        param.requires_grad = True
                else:
                    for param in network_list[key].parameters():
                        param.requires_grad = False
            loss_domain.backward(retain_graph=True)
            
            for key in network_list.keys():
                if "class_feature" in key or "task" in key:
                    for param in network_list[key].parameters():
                        param.requires_grad = True
                else:
                    for param in network_list[key].parameters():
                        param.requires_grad = False

        loss.backward()
        if self.bool_angle or self.bool_task:
            for key in network_list.keys():
                for param in network_list[key].parameters():
                    param.requires_grad = True

            self.optimizer_domain.step()
            if self.bool_task:
                self.optimizer_task.step()

        self.optimizer.step()

        return output

    def predict(self, x):
        return self.network(x)

    def predict_domain(self, x):
        return self.network_domain(x)

    def get_forward_model(self):
        forward_model = ForwardModel(self.network)
        return forward_model

    def predict_task(self, x):
        return self.classifier_task(x)