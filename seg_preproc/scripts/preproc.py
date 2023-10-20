#!/usr/bin/env python
import sys
import os
from typing import Optional
import cv2
import torch
import json
import numpy as np

# ROS dependencies
import rospy
import rospkg
from cv_bridge import CvBridge, CvBridgeError
import message_filters

# ROS messages
from sensor_msgs.msg import Image, CameraInfo

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import SegFormerTRT
from src.FastSAM.fastsam import FastSAM
from utils.utils import concatenating_map


class Preproc:
    def __init__(self,
                 segformer_trt_file: str,
                 segformer_taxonomy: str,
                 fastsam_pt_file: Optional[str] = None,
                 refinement_enabled: Optional[bool] = False,
                 blur_filter_enabled: Optional[bool] = False,
                 blur_threshold: Optional[int] = 20,
                 resolution: Optional[tuple] = (1280, 720),
                 lookup_table: Optional[list] = None
                 ) -> None:
        self.bridge = CvBridge()
        self.resolution = resolution
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            raise RuntimeError('CUDA is not available.')

        self.segformer_trt = SegFormerTRT(segformer_trt_file, segformer_taxonomy)
        self.fastsam_model = FastSAM(fastsam_pt_file) if refinement_enabled else None

        self.rgb_sub = message_filters.Subscriber('/rgb/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/depth_to_rgb/image_raw', Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 3000, 0.1,
                                                              allow_headerless=True)
        self.ts.registerCallback(self.preproc_callback)

        self.seg_pub = rospy.Publisher('/preproc/segmentation', Image, queue_size=10)
        self.depth_pub = rospy.Publisher('/preproc/depth', Image, queue_size=10)

        self.blur_filter_enabled = blur_filter_enabled
        self.blur_threshold = blur_threshold

        self.lookup_table = np.array(lookup_table) if lookup_table else None

    def refine_seg(self, img, seg):
        sam_out = self.fastsam_model(img, device=self.device, retina_masks=False, iou=0.7, conf=0.5, imgsz=1024,
                                     max_det=300)
        masks = sam_out[0].masks.data.cpu().detach().numpy().astype(bool)

        masks = torch.from_numpy(masks).to(self.device)
        seg = torch.from_numpy(seg).to(self.device)

        cls = torch.nn.functional.one_hot(seg.long(), 56).permute(0, 1, 2).float().to(self.device)

        for i in range(masks.shape[0]):
            mask = masks[i]
            seg_mask = seg * mask.int()
            vals, counts = seg_mask.unique(return_counts=True)
            non_zero = vals != 0
            vals, counts = vals[non_zero], counts[non_zero]
            for val, cnt in zip(vals, counts):
                cls[:, :, val] += (mask * cnt).int()

        # Normalization
        total_cnt = torch.sum(cls, dim=-1, keepdim=True)
        total_cnt[total_cnt == 0] = 1
        cls /= total_cnt

        cls = torch.argmax(cls, dim=-1).type(torch.int32).cpu().numpy()
        return cv2.resize(cls, self.resolution, interpolation=cv2.INTER_NEAREST)

    def preproc_callback(self, rgb: Image, depth: Image):
        rospy.loginfo("Image Received")
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
        except CvBridgeError as e:
            print(e)

        if self.blur_filter_enabled:
            laplacian_var = cv2.Laplacian(cv_rgb, cv2.CV_64F).var()
            print("laplacian_var:", laplacian_var)
            if laplacian_var < self.blur_threshold:
                print("too much motion blur, skip current frame")
                return

        sf_in = self.segformer_trt.preprocess_input(cv_rgb)
        sf_out = self.segformer_trt.inference(sf_in)
        seg = self.segformer_trt.postprocess_output(sf_out, (1024, 576))

        if self.fastsam_model:
            seg = self.refine_seg(cv_rgb, seg)
        else:
            seg = cv2.resize(seg, self.resolution, interpolation=cv2.INTER_NEAREST)

        if self.lookup_table is not None:
            seg = self.lookup_table[seg]

        seg_msg = self.bridge.cv2_to_imgmsg(cv2.convertScaleAbs(seg * 5), "8UC1")
        seg_msg.header = rgb.header
        self.seg_pub.publish(seg_msg)
        self.depth_pub.publish(depth)


def main():
    rospy.init_node("preproc")
    rospy.loginfo("node: preproc")

    rospkg_path = rospkg.RosPack().get_path('seg_preproc')

    with open(rospkg_path + '/resources/dms.json', 'r') as file:
        semantic_data = json.load(file)
    with open(rospkg_path + '/resources/map.json', 'r') as file:
        map_data = json.load(file)

    default_params = {
        '~resolution': (1280, 720),
        '~segformer_trt_file': rospkg_path + '/resources/trt_engine_8.6.1.trt',
        '~segformer_taxonomy': rospkg_path + '/resources/taxonomy.json',
        '~refinement_enabled': True,
        '~fastsam_pt_file': rospkg_path + '/resources/FastSAM-s.pt',
        '~blur_filter_enabled': False,
        '~blur_threshold': 20,
        '~lookup_table': concatenating_map(semantic_data, map_data)
    }
    params = {param.strip('~'): rospy.get_param(param, default) for param, default in default_params.items()}

    Preproc(**params)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()
