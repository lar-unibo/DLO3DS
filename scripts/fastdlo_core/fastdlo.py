import os, cv2
from fastdlo_core.core import Pipeline


class FASTDLO():

    def __init__(self, main_folder, mask_th = 127, ckpt_siam_name = "CP_similarity.pth", ckpt_seg_name = "CP_segmentation.pth", img_w = 640, img_h = 360):

        self.mask_th = mask_th

        checkpoint_siam = os.path.join(main_folder, "fastdlo_core/checkpoints/" + ckpt_siam_name)
        checkpoint_seg = os.path.join(main_folder, "fastdlo_core/checkpoints/" + ckpt_seg_name)

        self.fastdlo = Pipeline(checkpoint_siam=checkpoint_siam, checkpoint_seg=checkpoint_seg, img_w=img_w, img_h=img_h)



    def generete_output_splines_msg(self, paths):
        tck_array = []
        
        for _, p in paths.items():

            spline_extended = p["spline_extended"]
            t = spline_extended["t"]
            c = spline_extended["c"]
            k = spline_extended["k"]
            cx = c[1]
            cy = c[0]	
            tck_array.append([t, cx, cy, k])

        return tck_array



    def run(self, img, debug=False):
        splines, mask_output = self.fastdlo.run(img, mask_th=self.mask_th)

        splines = self.generete_output_splines_msg(splines)
        return splines, mask_output        

