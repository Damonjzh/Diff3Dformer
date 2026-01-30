"""
DETR model and criterion classes.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .transformer import build_transformer
from .position_encoding import build_position_encoding
from einops import rearrange, repeat
class mortality_ViT(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, transformer, position_embedding, max_num_cluster, withPosEmbedding, seq_pool,withLN=False,withEmbeddingPreNorm = False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.withLN = withLN
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.embedding_proj = nn.Linear(512,hidden_dim)
        # self.embedding_proj = nn.Linear(128, hidden_dim)
        self.position_embedding = position_embedding
        self.survival_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.withEmbeddingPreNorm = withEmbeddingPreNorm
        if not self.withLN:
            self.mlp_head = nn.Linear(hidden_dim, 1)
            self.mlp_head_MSE = nn.Linear(hidden_dim, 1)
        else:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1)
            )
            self.mlp_head_MSE = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1)
            )
        self.max_num_cluster = max_num_cluster
        self.survival_score_cluster = nn.Parameter(torch.randn(self.max_num_cluster+1))
        self.maxSurvivalTime = 13
        self.timeInterval = 0.1
        self.baseline_hazard_survival = nn.Parameter(torch.rand(int(self.maxSurvivalTime/self.timeInterval+1)))
        self.norm_Embedding = nn.LayerNorm(hidden_dim)
        self.withPosEmbedding=withPosEmbedding
        self.seq_pool=seq_pool

        #add
        # self.risk = nn.Parameter(torch.randn(3,64))
        self.risk = nn.Parameter(torch.randn(64))
        print(0)

    def predictSurvivalTime(self, survivalRisk):
        cumsum_baseline_hazard = torch.cumsum(self.baseline_hazard_survival, 0)
        baseline_survival = torch.exp(-cumsum_baseline_hazard*self.timeInterval)
        survival_t = torch.pow(repeat(baseline_survival,'t -> t b', b = len(survivalRisk)), repeat(torch.exp(survivalRisk),'b -> t b', t = len(baseline_survival)))
        predictedSurvivalTime = torch.sum(survival_t,dim=0)* self.timeInterval 
        return predictedSurvivalTime  

    def forward(self, patientEmbedding, pos, keyPaddingMask, cluster):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        patientEmbedding = self.embedding_proj(patientEmbedding)
        if not self.withEmbeddingPreNorm:
            x = patientEmbedding
        else:
            x=self.norm_Embedding(patientEmbedding)     
        b, n, _ = x.shape
        device = cluster.device

        survival_token = repeat(self.survival_token, '() n d -> b n d', b = b)
        x = torch.cat((survival_token, x), dim=1)
        cluster_survival_token = self.max_num_cluster*torch.ones([b,1],dtype=torch.int64).to(device)
        cluster = torch.cat((cluster_survival_token, cluster), dim=1)
        tokenPadding = torch.zeros(b,1).to(device)
        keyPaddingMask = torch.cat((tokenPadding, keyPaddingMask), dim=1)
        if self.withPosEmbedding:
            emb = self.position_embedding(pos[:,:,0],pos[:,:,1],pos[:,:,2])
            x = x+emb
        x = x.transpose(0, 1)
# temp type conversion
        cluster = cluster.long()
        keyPaddingMask = keyPaddingMask.bool()
        x, A = self.transformer(x, mask=None, src_key_padding_mask=keyPaddingMask,clusters = cluster, pos=pos)

        # survival token outputs survival time, others output survival risk
        if self.seq_pool:
            keyPaddingMask = rearrange(keyPaddingMask,'b n -> n b')
            keyPaddingMask = repeat(keyPaddingMask, 'n b -> n b d', d = 1)
            x = x[1:x.shape[0]]

            survival_score = self.mlp_head(x)
            survivalRisk = torch.zeros(survival_score.shape[1], device=device)
            mean_survival_score_batchsize = torch.zeros(survival_score.shape[1],64, device=device)
            percentage_clusters_batchsize = torch.zeros(survival_score.shape[1],64, device=device)
            cluster = cluster[:,1:cluster.shape[1]]
            for patient_idx in range(survival_score.shape[1]):
                valid_indexes = cluster[patient_idx] != 64
                patient_survival_score = survival_score[valid_indexes, patient_idx, :]
                patient_cluster = cluster[patient_idx,valid_indexes]

                cluster_counts = torch.bincount(patient_cluster, minlength = 64)
                total_clusters = torch.sum(cluster_counts[:64])
                percentage_clusters = cluster_counts[: 64] / total_clusters
                cluster_scores = {cluster:[] for cluster in range(64)}
                percentage_clusters_batchsize[patient_idx] = percentage_clusters

                for score, cluster_idx in zip(patient_survival_score, patient_cluster):
                    if cluster_idx.item() == 64:
                        continue
                    cluster_scores[cluster_idx.item()].append(score.item())
                mean_survival_score = {cluster: torch.mean(torch.tensor(scores, device=device)) for cluster, scores in
                                       cluster_scores.items()}
                mean_survival_score_array = [mean_survival_score[key].item() if not torch.isnan(mean_survival_score[key]) else 0 for key in mean_survival_score]
                mean_survival_score_batchsize [patient_idx] = torch.asarray(mean_survival_score_array)
                for i in range(64):
                    if percentage_clusters[i] == 0:
                        continue
                    survivalRisk[patient_idx] += percentage_clusters[i] * self.risk[i] * mean_survival_score[i]

        else:
            survival_score = self.mlp_head(x[1:x.shape[0]]).squeeze(2) 
            survivalRisk = (survival_score).mean(dim = 0)

        return survivalRisk,survival_score,A,self.risk,mean_survival_score_batchsize, percentage_clusters_batchsize

class SetCriterion(nn.Module):
    def __init__(self):
        super(SetCriterion, self).__init__()
        self.loss_dict = {}
        self.weight_dict = {'neg_likelihood':1}
        self.survival_score_Loss = nn.MSELoss()

    def forward(self, outputs: Tensor, survivalTimes: Tensor, Dead: Tensor) -> Tensor:
        """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.
        We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
        where h = exp(log_h) are the hazards and R is the risk set, and d is event.
        We just compute a cumulative sum, and not the true Risk sets. This is a
        limitation, but simple and fast.
        """
        survivalRisk,survival_score,A = outputs
        idx = survivalTimes.sort(descending=True)[1]
        events = Dead[idx]
        try:
            log_h = survivalRisk[idx]
            neg_likelihood=self.cox_ph_loss_sorted(log_h, events)
            self.loss_dict = {'neg_likelihood':neg_likelihood}
        except:
            print(idx.shape)

        return self.loss_dict

    def cox_ph_loss_sorted(self, log_h: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
        """Requires the input to be sorted by descending duration time.
        See DatasetDurationSorted.
        We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
        where h = exp(log_h) are the hazards and R is the risk set, and d is event.
        We just compute a cumulative sum, and not the true Risk sets. This is a
        limitation, but simple and fast.
        """
        if events.dtype is torch.bool:
            events = events.float()
        events = events.view(-1)
        log_h = log_h.view(-1)
        gamma = log_h.max()
        log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
        return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum()+eps)

def build_mortality_ViT(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223

    device = torch.device(args.device)
    max_num_cluster = args.max_num_cluster

    transformer = build_transformer(args)

    position_embedding = build_position_encoding(args)
    model = mortality_ViT(transformer,position_embedding,max_num_cluster,args.withPosEmbedding, seq_pool=args.seq_pool, withLN=args.withLN, withEmbeddingPreNorm=args.withEmbeddingPreNorm)
    criterion = SetCriterion()
    criterion.to(device)
    return model, criterion


def build_mortality_ViT_new(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223

    device = torch.device(args.device)
    max_num_cluster = args.max_num_cluster

    transformer = build_transformer(args)

    position_embedding = build_position_encoding(args)
    model = mortality_ViT(transformer,position_embedding,max_num_cluster,args.withPosEmbedding, seq_pool=args.seq_pool, withLN=args.withLN, withEmbeddingPreNorm=args.withEmbeddingPreNorm)
    criterion = SetCriterion()
    criterion.to(device)
    lf_bce = nn.BCEWithLogitsLoss().cuda()
    return model, lf_bce

from torch.autograd import Variable
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        alpha = torch.tensor([alpha, 1])
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss