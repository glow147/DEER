import lightning as L 
import torch
import torchvision.ops as ops
from torch import nn
from torchvision.models import resnet50
from torch.nn.utils.rnn import pad_sequence

from models.backbone import vovnet
from models.loss import BalanceCrossEntropyLoss, DiceLoss,BalanceL1Loss
from models.deformable_transformer import DEER_Encoder,LocationHead,DEER_Decoder
from models.extract import process_text_regions
from models import utils
from transformers import AutoTokenizer

class DEER(L.LightningModule) :
    def __init__(self, config):
        super(DEER, self).__init__()
        self.config = config
        self.hidden_dim = 256
        self.batch_size = self.config.MODEL.PARAMS.BATCH_SIZE
        self.img_size = self.config.DATA.IMAGE_SIZE

        self.backbone = vovnet.VoVNet('V-39-eSE', 3, ["stage2", "stage3", "stage4", "stage5"])
        self.backbone.load_state_dict(torch.load(self.config.MODEL.BACKBONE.PRETRAINED))    

        # Encoder
        self.encoder = DEER_Encoder(d_model=self.hidden_dim,dim_feedforward=1024,dropout=0.1,activation="gelu",
                                    num_feature_levels=4,nhead=8,enc_n_points=4,num_encoder_layers=6)
        
        # location head
        self.location_head = LocationHead(self.hidden_dim)

        # Decoder 
        self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

        self.decoder = DEER_Decoder(max_len=self.config.DATA.MAX_LEN,d_model=self.hidden_dim,dim_feedforward=1024,dropout=0.1,
                                    activation="gelu",num_feature_levels=4,nhead=8,dec_n_points=4,num_decoder_layers=6,tokenizer=self.tokenizer)
        
        
        # Loss
        self.char_loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.dice_loss = DiceLoss()
        self.L1_loss = BalanceL1Loss()
        self.bce_loss = BalanceCrossEntropyLoss()

        self.encoder.apply(self.weights_init)
        self.decoder.apply(self.weights_init)

    def weights_init(self, m) -> None:
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif classname.find('Linear') != -1:
            nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif classname.find('GroupNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
    
    def forward(self, image, labels, pref_noise):
        # Feature extraction from backbone
        feature_maps = self.backbone(image)
        
        # Deformable DETR encoder
        refined_features,level_start_index,spatial_shapes,valid_ratios = self.encoder(feature_maps)

        # Location Head
        db_maps = self.location_head(refined_features,feature_maps,level_start_index)

        # DETR text decoder
        outputs_class = self.decoder(labels,pref_noise,refined_features,spatial_shapes,level_start_index,valid_ratios,None,True)
        
        return db_maps, outputs_class

    @torch.inference_mode()
    def inference(self, image):
        '''
        Example Draw Result
	# for filename, image in zip(filenames, images):
        #     center_point, pred_word = self.inference(image)
        #     utils.draw_pred_text(filename, image.cpu(), center_point, pred_word)
        '''
        # Feature extraction from backbone
        refined_features, spatial_shapes, level_start_index, valid_ratios, center_points = self.get_text_region(image)
      
        if len(center_points) == 0:
            return None, None

        pred_words = self.get_text(refined_features, spatial_shapes, level_start_index, valid_ratios, center_points[None, ...])

        return center_points, pred_words[0]

    def training_step(self, batch, batch_idx):
        image = batch["image"].to(self.device)
        masks = batch["masks"].to(self.device, dtype=torch.bool)
        gts = batch["gts"].to(self.device, dtype=torch.float)
        thresh_map = batch['thresh_map'].to(self.device)
        thresh_mask = batch['thresh_mask'].to(self.device)

        pref_noise = batch["pref_noise"].to(self.device)
        labels = batch["label"]

        tokenized_labels = pad_sequence([self.text_encode(label) for label in labels],batch_first=True, padding_value=self.tokenizer.pad_token_id)
        tokenized_labels = tokenized_labels.to(self.device)

        db_maps, outputs_class = self(image, tokenized_labels[:,:,:-1], pref_noise)

        loss, char_loss, prob_loss, binary_loss, thresh_loss = self.calculate_loss(tokenized_labels, gts, masks, db_maps, thresh_map, thresh_mask, outputs_class)
        self.log('train_loss', loss, on_step=True,on_epoch=True, prog_bar=True,sync_dist=True,batch_size=self.batch_size)
        self.log('char_loss', char_loss, on_step=True, on_epoch=True, sync_dist=True,prog_bar=True,batch_size=self.batch_size)
        self.log('prob_loss', prob_loss, on_step=True, prog_bar=True,sync_dist=True,batch_size=self.batch_size)
        self.log('binary_loss', binary_loss, on_step=True, prog_bar=True,sync_dist=True,batch_size=self.batch_size)
        self.log('thresh_loss', thresh_loss, on_step=True, prog_bar=True,sync_dist=True,batch_size=self.batch_size)

        return {'loss': loss}

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        image = batch["image"]
        labels = batch["label"]
        bboxes = batch["bboxes"]

        feature_maps = self.backbone(image)
        # Deformable DETR encoder
        refined_features, level_start_index, spatial_shapes, valid_ratios = self.encoder(feature_maps)

        # Location Head
        db_maps = self.location_head(refined_features, feature_maps, level_start_index)

        pred_boxes, pred_maps = process_text_regions(db_maps["binary"], threshold=0.7, dilation_factor=2.0)

        recall_tp, recall_fn, gtNums, pred_refs = self.match_predict(pred_boxes, bboxes, image.shape[2], image.shape[3], 0.5)

        pred_words = self.get_text(refined_features, spatial_shapes, level_start_index, valid_ratios, pred_refs)
        
        precision_tp, precision_fp = 0, 0
        for i in range(len(gtNums)):
            for j, pred_word in enumerate(pred_words[i]):
                if pred_word == labels[i][gtNums[i][j]]:
                    precision_tp += 1
                else:
                    precision_fp += 1

        self.precision_tp += precision_tp
        self.precision_fp += precision_fp
        self.recall_tp += recall_tp
        self.recall_fn += recall_fn

    def on_validation_epoch_start(self):
        self.precision_tp = 0
        self.precision_fp = 0
        self.recall_tp = 0
        self.recall_fn = 0

    def on_validation_epoch_end(self):
        precision = self.precision_tp / max(1, self.precision_tp + self.precision_fp)
        recall = self.recall_tp / max(1, self.recall_tp + self.recall_fn)

        # Micro F1 score
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        # Log the micro F1 score
        self.log('val_precision', precision, prog_bar=True, sync_dist=True)
        self.log('val_recall', recall, prog_bar=True, sync_dist=True)
        self.log('val_f1', f1_score, prog_bar=True, sync_dist=True)
        
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def test_step(self, batch, batch_idx):
        images = batch["image"]
        labels = batch["label"]
        bboxes = batch["bboxes"]
        filenames = batch["filename"]

        feature_maps = self.backbone(images)
        # Deformable DETR encoder
        refined_features, level_start_index, spatial_shapes, valid_ratios = self.encoder(feature_maps)

        # Location Head
        db_maps = self.location_head(refined_features, feature_maps, level_start_index)

        pred_boxes, pred_maps = process_text_regions(db_maps["binary"], threshold=0.7, dilation_factor=2.0)

        recall_tp, recall_fn, gtNums, pred_refs = self.match_predict(pred_boxes, bboxes, images.shape[2], images.shape[3], 0.5)

        pred_words = self.get_text(refined_features, spatial_shapes, level_start_index, valid_ratios, pred_refs)
        
        precision_tp, precision_fp = 0, 0
        for i in range(len(gtNums)):
            for j, pred_word in enumerate(pred_words[i]):
                if pred_word == labels[i][gtNums[i][j]]:
                    precision_tp += 1
                else:
                    precision_fp += 1

        self.precision_tp += precision_tp
        self.precision_fp += precision_fp
        self.recall_tp += recall_tp
        self.recall_fn += recall_fn

    def on_test_epoch_start(self):
        self.precision_tp = 0
        self.precision_fp = 0
        self.recall_tp = 0
        self.recall_fn = 0


    def on_test_epoch_end(self):
        precision = self.precision_tp / max(1, self.precision_tp + self.precision_fp)
        recall = self.recall_tp / max(1, self.recall_tp + self.recall_fn)

        # Micro F1 score
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        self.log('precision', precision, prog_bar=True, sync_dist=True)
        self.log('recall', recall, prog_bar=True, sync_dist=True)
        self.log('f1', f1_score, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        world_size = torch.distributed.get_world_size()
        optim = getattr(torch.optim, self.config.MODEL.OPTIMIZER)
        optimizer = optim(self.parameters(), lr=self.config.MODEL.SCHEDULER_PARAMS.max_lr, weight_decay=self.config.MODEL.PARAMS.WEIGHT_DECAY)
        
        if not self.config.MODEL.SCHEDULER:
            return optimizer
        
        elif hasattr(torch.optim.lr_scheduler, self.config.MODEL.SCHEDULER):
            scheduler = getattr(torch.optim.lr_scheduler, self.config.MODEL.SCHEDULER)
            scheduler = scheduler(optimizer, **self.config.MODEL.SCHEDULER_PARAMS)

        elif hasattr(utils, self.config.MODEL.SCHEDULER):
            scheduler = getattr(utils, self.config.MODEL.SCHEDULER)
            scheduler = scheduler(
                            optimizer=optimizer,
                            first_cycle_steps=self.config.MODEL.PARAMS.MAX_STEPS * (32 // (self.config.MODEL.PARAMS.BATCH_SIZE * world_size)), 
                            cycle_mult=1.,             
                            max_lr=self.config.MODEL.SCHEDULER_PARAMS.max_lr,                
                            min_lr=self.config.MODEL.SCHEDULER_PARAMS.min_lr,                
                            warmup_steps=10000 * (32 // (self.config.MODEL.PARAMS.BATCH_SIZE * world_size)),
                            gamma=1.          
                        )
        else:
            raise ModuleNotFoundError
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # step 기준으로 스케줄러 업데이트
                'frequency': 1       # 매 스텝마다 업데이트
            }
        }

    def calculate_loss(self, labels, gt, gt_mask, maps, thresh_map, thresh_mask, outputs_class):
        char_loss = self.char_loss(outputs_class.contiguous().view(-1,self.tokenizer.vocab_size), labels[:,:,1:].contiguous().view(-1))

        prob_loss, _ = self.bce_loss(maps["binary"], gt, gt_mask)
        thresh_loss = self.L1_loss(maps["thresh"],thresh_map,thresh_mask)
        binary_loss = self.dice_loss(maps["thresh_binary"], gt, gt_mask)

        loss = prob_loss + binary_loss + thresh_loss + char_loss

        return loss, char_loss, prob_loss, binary_loss, thresh_loss
    
    def text_encode(self, label):
        tokens = self.tokenizer.batch_encode_plus(label, truncation=True, padding='max_length', max_length=self.config.DATA.MAX_LEN, return_tensors="pt")

        return tokens['input_ids']
    
    def match_predict(self, pred_boxes, gt_boxes, height, width, iou_threshold=0.5):
        gtNums = []
        pred_refs = []  # predict reference points
        tp, fn = 0, 0

        for preds, gts in zip(pred_boxes, gt_boxes):
            matched_preds = set()
            matched_gts = set()

            # If no predictions
            if len(preds) == 0:
                pred_refs.append([])
                gtNums.append([])
                fn += len(gts)
                continue

            # Convert GT boxes to xyxy format
            gts = ops.box_convert(gts, "xywh", "xyxy")

            # Compute IoU
            iou_matrix = ops.box_iou(preds, gts)

            # Flatten the IoU matrix to a list of (pred_idx, gt_idx, iou)
            pred_indices, gt_indices = torch.where(iou_matrix >= iou_threshold)
            ious = iou_matrix[pred_indices, gt_indices]

            # Sort matches by IoU in descending order
            sorted_indices = torch.argsort(ious, descending=True)
            pred_indices = pred_indices[sorted_indices]
            gt_indices = gt_indices[sorted_indices]
            ious = ious[sorted_indices]

            gtNum = []
            centers = []

            for pred_idx, gt_idx, iou in zip(pred_indices, gt_indices, ious):
                pred_idx_item = pred_idx.item()
                gt_idx_item = gt_idx.item()

                if pred_idx_item not in matched_preds and gt_idx_item not in matched_gts:
                    matched_preds.add(pred_idx_item)
                    matched_gts.add(gt_idx_item)
                    tp += 1
                    gtNum.append(gt_idx_item)

                    # Compute center point of the predicted box
                    pred_box = preds[pred_idx]
                    center_point = (pred_box[:2] + pred_box[2:]) / 2  # x1, y1, x2, y2
                    center_point /= torch.tensor([width, height], device=self.device)
                    centers.append(center_point)

            # Count unmatched GT boxes as false negatives
            fn += len(gts) - len(matched_gts)

            if gtNum:
                gtNums.append(torch.tensor(gtNum, device=self.device))
            else:
                gtNums.append(torch.tensor([], device=self.device))

            if centers:
                pred_refs.append(torch.stack(centers))
            else:
                pred_refs.append([])

        return tp, fn, gtNums, pred_refs

    def get_text_region(self, image):
        feature_maps = self.backbone(image[None, ...])

        refined_features, level_start_index, spatial_shapes, valid_ratios = self.encoder(feature_maps)

        db_maps = self.location_head(refined_features, feature_maps, level_start_index)

        pred_boxes, pred_maps = process_text_regions(db_maps["binary"], threshold=0.7, dilation_factor=2.0)

        pred_boxes = pred_boxes[0]
        if len(pred_boxes) == 0:
            return refined_features, spatial_shapes, level_start_index, valid_ratios, []
        
        center_points = pred_boxes[:, :2] + (pred_boxes[:, 2:] - pred_boxes[:, :2]) / 2

        center_points[:, 0] = center_points[:, 0] / image.shape[2]
        center_points[:, 1] = center_points[:, 1] / image.shape[1]

        return refined_features, spatial_shapes, level_start_index, valid_ratios, center_points

    def get_text(self, refined_features, spatial_shapes, level_start_index, valid_ratios, pred_refs):
        predict_words = []
        max_len = self.config.DATA.MAX_LEN

        for i, pred_ref in enumerate(pred_refs):
            if len(pred_ref) == 0:
                predict_words.append([])
                continue

            pred_ref = pred_ref.unsqueeze(0)  
            batch_size, num_queries = pred_ref.shape[:2]

            predict_word = torch.full(
                (batch_size, num_queries, max_len),
                fill_value=self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=self.device
            )
            predict_word[:, :, 0] = self.tokenizer.cls_token_id 

            for step in range(1, max_len):
                outputs = self.decoder(
                    predict_word[:, :, :step],
                    pred_ref,
                    refined_features[i][None],
                    spatial_shapes,
                    level_start_index,
                    valid_ratios[i][None],
                    None,
                    False
                )

                logits = outputs[:, :, -1, :]  
                pred_tokens = torch.argmax(logits, dim=-1) 
                predict_word[:, :, step] = pred_tokens 

            decoded_words = [
                self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for tokens in predict_word[0]
            ]
            predict_words.append(decoded_words)

        return predict_words