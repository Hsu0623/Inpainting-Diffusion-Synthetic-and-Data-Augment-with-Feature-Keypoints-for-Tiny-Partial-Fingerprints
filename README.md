# Inpainting-Diffusion-Synthetic-and-Data-Augment-with-Feature-Keypoints-for-Tiny-Partial-Fingerprints
### Overall 
![overview](https://github.com/user-attachments/assets/05c924c9-493a-45e1-b823-e12ddb72faef)

### realease datasets:
![image](https://github.com/Hsu0623/Inpainting-Diffusion-Synthetic-and-Data-Augment-with-Feature-Keypoints-for-Tiny-Partial-Fingerprints/assets/67309197/d9241515-6109-46fc-a199-c39ecd4f5f6f)

The total released datasets is described:
We have three subdatasets: June, May_01, May_02. In the following link, May_01 is named May_noLabel and May_02 is named May_withLabel.
 
DDIM_fake_matched (which are the synthetic fingerprint from the diffusion sampling with label)：
http://gofile.me/7p5O4/DcJINvNVU

ddim_unmatched：(which are the synthetic fingerprint from the diffusion sampling without label):
http://gofile.me/7p5O4/BsvKGEFhH

inpaint_with_feature_keypoints (which are synthetic by inpaintin diffusion with feature kepoints, with Label is inpaint_matched_v1, and without Label is virtual_ddim_unmatched.)：
http://gofile.me/7p5O4/OvdUhKSN7

### How to Inference
1. download the pretrained weight
2. Create feature keypoint mask
3. modified the yml file for your path (gt_path, mask_path, srs, lsr, gts, gt_keep_masks)
4. $ python test.py
   

### How to Train on your own datasets
please refer https://github.com/openai/guided-diffusion?tab=readme-ov-file

### citation
1.Dhariwal, Prafulla, and Alexander Nichol. "Diffusion models beat gans on image synthesis." Advances in neural information processing systems 34 (2021): 8780-8794.(https://github.com/openai/guided-diffusion?tab=readme-ov-file)

2.Lugmayr, Andreas, et al. "Inpainting using denoising diffusion probabilistic models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.(https://github.com/andreas128/RePaint?tab=readme-ov-file)


