from .detector3d_template import Detector3DTemplate

import random
import time
class NV3D(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        """
        module_topology = [
            'vfe', 'vector_fe', 'backbone_3d', 'map_to_bev_module',
            'backbone_2d', 'dense_head', 'roi_head'
        ]
        
        x = random.random()
        
        s1 = time.time()
        for cur_module, name_module in zip(self.module_list,module_topology):
            s2 = time.time()
            batch_dict = cur_module(batch_dict)
            e2 = time.time()
            t2 = e2 - s2
            print(f"Time ({name_module}): {t2} seconds")
            
        e1 = time.time()
        t1 = e1 - s1
                
        print(f"Time (Total): {t1} seconds")
        print("--------------------")
            
        if x > .8: print("".shape)
        
        """
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        
        
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn
        
        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d
            
        return loss, tb_dict, disp_dict
