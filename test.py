import os
import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
from model.M import MINet
from data1.util import get_img_patches, merge_pred_patches
from data1 import create_dataset
import torch.nn.functional as F

def save_sample(img_path, msk, msk_pred, save_dir, name=''):
    img = cv2.imread(img_path)
    if img is None:
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = msk.shape
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    msk = msk.astype(np.uint8)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(img / 255.)
    plt.title('Input')

    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(msk * 255, cmap='gray')
    plt.title('GT')

    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(msk_pred * 255, cmap='gray')
    plt.title('Pred')

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{name}.png"), bbox_inches='tight', dpi=300)
    plt.close()

channel_means = [0.598, 0.584, 0.565]
channel_stds = [0.104, 0.103, 0.103]

class ImgToTensor(object):
    def __call__(self, img):
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(channel_means, channel_stds)
        ])
        return tf(img)

if __name__ == '__main__':
    config_path = r'\configs\crackmap.yaml'
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    SAVE_VIS_DIR = "./vis_results_final"
    os.makedirs(SAVE_VIS_DIR, exist_ok=True)

    class Args:
        def __init__(self, phase, dataset_path, load_width, load_height):
            self.phase = phase
            self.dataset_path = dataset_path
            self.load_width = load_width
            self.load_height = load_height

    data_path = config['path_to_testdata']
    test_args = Args(phase='test', dataset_path=data_path, load_width=256, load_height=256)
    test_dataset = create_dataset(test_args)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    Net = MINet().to(device)
    weight_dict = torch.load(config['saved_model'], map_location=device)
    if 'model_weights' in weight_dict:
        Net.load_state_dict(weight_dict['model_weights'])
    else:
        Net.load_state_dict(weight_dict)
    Net.eval()

    patch_totensor = ImgToTensor()

    with torch.no_grad():
        for itter, batch in enumerate(tqdm(test_loader)):

            img_tensor = batch['image']
            img = img_tensor.squeeze(0).cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            img = (img * channel_stds + channel_means) * 255
            img = np.clip(img, 0, 255).astype(np.uint8)

            msk = batch['label'] if 'label' in batch else batch['mask']
            msk = msk.cpu().numpy()[0, 0]

            img_path = batch['A_paths'][0] if 'A_paths' in batch else batch['img_path'][0]

            patches, patch_locs = get_img_patches(img)
            preds = []
            for patch in patches:
                if len(patch.shape) == 2:
                    patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)

                patch_pil = Image.fromarray(patch)
                patch_tensor = patch_totensor(patch_pil).unsqueeze(0).to(device)
                outputs = Net(patch_tensor)  # (out1, out2, out3, out4, out5)

                final_out = outputs[-1]


                final_out = F.interpolate(
                    final_out,
                    size=patch_tensor.shape[2:],
                    mode='bilinear',
                    align_corners=True
                )

                pred_prob = torch.sigmoid(final_out)
                pred_binary = (pred_prob > 0.5).float()

                preds.append(pred_binary.detach().cpu().numpy()[0, 0])

            mskp = merge_pred_patches(img, preds, patch_locs)


            save_sample(img_path, msk, mskp, SAVE_VIS_DIR, name=str(itter + 1))
