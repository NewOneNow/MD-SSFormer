import os
# import cv2
import torch
from networks.runet import RUNet
from networks.MyNet import R2U_Net,NestedUNet

from networks.MyNet import AttU_Net,R2AttU_Net
from networks.attpunetattxin import AttpU_Net
from networks.res_unet_plus import ResUnetPlusPlus
from networks.attplusunet import AttpUdeep_Net
from networks.attpunetattxin64 import AttpU64_Net
from networks.maunet1 import MAU_Net1
from networks.mau_net import MAU_Net
from networks.frequencynet import frequencynet
from networks.ERSUnet import ERSUnet
# from networks.Attpaspp import AttpasppU_Net
from networks.UNet import UNet
import shutil
import utils.image_transforms as joint_transforms
from torch.utils.data import DataLoader
import utils.transforms as extended_transforms
from datasets import bladder3 as bladder
from utils.loss import *
from utils.metrics import diceCoeffv2,jaccardv2,accuracy, sp,se, HausdorffLoss, specificity
from networks.u_net import Baseline
from tqdm import tqdm
from networks.flashUnet import flashUNet
from networks.BATFormer import C2FTransformer
crop_size = 512
#/raid0/name/chengdongxu/lizenghui/LiTS
# val_path = r'/raid0/name/chengdongxu/lizenghui/LiTSdatapng/tumor'
# val_path = r'D:\pythonProject\pytorch-medical-image-segmentation-master\datasets'
val_path = r'datasets/BUSI'

# val_path = r'..\media/Datasets/Bladder/raw_data'
center_crop = joint_transforms.CenterCrop(crop_size)
val_input_transform = extended_transforms.NpyToTensor()
target_transform = extended_transforms.MaskToTensor()

val_set = bladder.Dataset(val_path, 'val', 1,
                              joint_transform=None, transform=val_input_transform, center_crop=center_crop,
                              target_transform=target_transform)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

palette = bladder.palette
num_classes = bladder.num_classes

# net = Baseline(img_ch=1, num_classes=num_classes, depth=2).cuda()
# net = patchUNet().cuda()
net = UNet().cuda()
# net = AttpUdeep_Net(n_channels=1, n_classes=2).cuda()
# net = AttpU64_Net(n_channels=1, n_classes=2).cuda()
# net = ERSUnet().cuda()
# net = AttpasppU_Net(n_channels=1, n_classes=2).cuda()
# net = NestedUNet().cuda()
# net = frequencynet().cuda()

# net = R2AttU_Net(n_channels=1, n_classes=2).cuda()

# Surface
net.load_state_dict(torch.load(r"Zhou/checkpoint/UNetOnBUSI_depth=5_fold_1_bce&edgeBilinearDice_685653.pth"))
net.eval()
def auto_val(net):
    # 效果展示图片数
    # hdLoss = HausdorffLoss()
    jaccards = 1
    dices = 1
    precisions = 1
    ses = 1
    accuracys = 1
    sps = 1
    # hd = 1
    class_dices = np.array([0] * (num_classes - 1), dtype=np.float64)
    class_jaccards = np.array([0] * (num_classes - 1), dtype=np.float64)
    class_precisions = np.array([0] * (num_classes - 1), dtype=np.float64)
    class_ses = np.array([0] * (num_classes - 1), dtype=np.float64)
    class_accuracys = np.array([0] * (num_classes - 1), dtype=np.float64)
    class_sps = np.array([0] * (num_classes - 1), dtype=np.float64)
    # class_hd = np.array([0] * (num_classes - 1), dtype=np.float)


    # save_path = r'D:\pythonProject\pytorch-medical-image-segmentation-master\validate\results'
    save_path = r'Zhou/results'
    if os.path.exists(save_path):
        # 若该目录已存在，则先删除，用来清空数据
        shutil.rmtree(os.path.join(save_path))
    img_path = os.path.join(save_path, 'images')
    pred_path = os.path.join(save_path, 'pred')
    gt_path = os.path.join(save_path, 'gt')
    os.makedirs(img_path)
    os.makedirs(pred_path)
    os.makedirs(gt_path)

    val_dice_arr = []
    val_jaccard_arr = []
    val_precision_arr = []
    val_se_arr = []
    val_accuracy_arr = []
    val_sp_arr = []
    # val_hd_arr = []

    for (input, mask), file_name in tqdm(val_loader):
        file_name = file_name[0].split('.')[0]
        file_name = file_name[21:]
        X = input.cuda()
        pred = net(X)
        pred = torch.sigmoid(pred)
        pred = pred.cpu().detach()

        pred[pred < 0.5] = 0
        pred[np.logical_and(pred > 0.5, pred == 0.5)] = 1


        # gt

        gt = helpers.onehot_to_mask(np.array(mask.squeeze()).transpose([1, 2, 0]), palette)
        gt = helpers.array_to_img(gt)

        # 原图
        m1 = np.array(input.squeeze())
        m1 = helpers.array_to_img(m1.transpose([2, 1, 0]))

        # pred
        save_pred = helpers.onehot_to_mask(np.array(pred.squeeze()).transpose([1, 2, 0]), palette)
        save_pred_png = helpers.array_to_img(save_pred)

        # png格式
        m1.save(os.path.join(img_path, file_name + '.png'))
        gt.save(os.path.join(gt_path, file_name + '.png'))
        save_pred_png.save(os.path.join(pred_path, file_name + '.png'))

        class_dice = []
        class_jaccard = []
        class_precision = []
        class_se = []
        class_accuracy = []
        class_sp = []
        # class_hd = []

        for i in range(1, num_classes):
            class_dice.append(diceCoeffv2(pred[:, i:i + 1, :], mask[:, i:i + 1, :]))
            class_jaccard.append(jaccardv2(pred[:, i:i + 1, :], mask[:, i:i + 1, :]))
            class_precision.append(precision(pred[:, i:i + 1, :], mask[:, i:i + 1, :]))
            class_se.append(se(pred[:, i:i + 1, :], mask[:, i:i + 1, :]))
            class_accuracy.append(accuracy(pred[:, i:i + 1, :], mask[:, i:i + 1, :]))
            class_sp.append(specificity(pred[:, i:i + 1, :], mask[:, i:i + 1, :]))
            # class_hd.append(hdLoss(pred[:, i:i + 1, :], mask[:, i:i + 1, :]))

        mean_dice = sum(class_dice) / len(class_dice)
        mean_jaccard = sum(class_jaccard) / len(class_jaccard)
        mean_precision = sum(class_precision) / len(class_precision)
        mean_se = sum(class_se) / len(class_se)
        mean_accuracy = sum(class_accuracy) / len(class_accuracy)
        mean_sp = sum(class_sp) / len(class_sp)
        # mean_hd = sum(class_hd) / len(class_hd)

        val_jaccard_arr.append(class_jaccard)
        val_dice_arr.append(class_dice)
        val_precision_arr.append(class_precision)
        val_se_arr.append(class_se)
        val_accuracy_arr.append(class_accuracy)
        val_sp_arr.append(class_sp)
        # val_hd_arr.append(class_hd)

        jaccards += mean_jaccard
        dices += mean_dice
        precisions += mean_precision
        ses += mean_se
        accuracys += mean_accuracy
        sps += mean_sp
        # hd += mean_hd

        class_dices += np.array(class_dice)
        class_jaccard += np.array(class_jaccard)
        class_precisions += np.array(class_precision)
        class_ses += np.array(class_se)
        class_accuracys += np.array(class_accuracy)
        class_sps += np.array(class_sp)
        # class_hd += np.array(class_hd)

        # print('mean_dice: {:.4} - dice_tumor: {:.4} - mean_jaccard: {:.4} - jaccard_tumor: {:.4} - mean_acc: {:.4} - acc_tumor: {:.4}'
        #           .format(mean_dice, class_dice[0], mean_jaccard, class_jaccard[0], mean_accuracy, class_accuracy[0]))
    val_mean_dice = dices / (len(val_loader) / 1)
    val_mean_jaccard = jaccards / (len(val_loader) / 1)
    val_mean_precision = precisions / (len(val_loader) / 1)
    val_mean_se = ses / (len(val_loader) / 1)
    val_mean_accuracy = accuracys / (len(val_loader) / 1)
    val_mean_sp = sps / (len(val_loader) / 1)
    # val_mean_hd = hd / (len(val_loader) / 1)

    val_class_dice = class_dices / (len(val_loader) / 1)
    val_class_jaccard = class_jaccards / (len(val_loader) / 1)
    val_class_precision = class_precisions / (len(val_loader) / 1)
    val_class_se = class_ses / (len(val_loader) / 1)
    val_class_accuracy = class_accuracys / (len(val_loader) / 1)
    val_class_sp = class_sps / (len(val_loader) / 1)
    # val_class_hd = class_hd / (len(val_loader) / 1)

    # print('Val mean_dice: {:.4} - dice_tumor: {:.4} - Val mean_jaccard: {:.4} - jaccard_tumor: {:.4} - mean_pre:{:.4} - pre_tumor:{:.4} - mean_se:{:.4} - se_tumor:{:.4} - mean_sp:{:.4} - sp_tumor:{:.4}  - mean_hd:{:.4} - hd_tumor:{:.4}'
    #       .format(val_mean_dice, val_class_dice[0], val_mean_jaccard, val_class_jaccard[0], val_mean_precision, val_class_precision[0], val_mean_se, val_class_se[0], val_mean_sp, val_class_sp[0], val_mean_hd, val_class_hd[0]))
    print(
        'Val mean_dice: {:.4} - dice_tumor: {:.4} - Val mean_jaccard: {:.4} - jaccard_tumor: {:.4} - mean_pre:{:.4} - pre_tumor:{:.4} - mean_se:{:.4} - se_tumor:{:.4} - mean_accuracy:{:.4} - accuracy_tumor:{:.4} - mean_sp:{:.4} - sp_tumor:{:.4}'
        .format(val_mean_dice, val_class_dice[0], val_mean_jaccard, val_class_jaccard[0], val_mean_precision,
                val_class_precision[0], val_mean_se, val_class_se[0],  val_mean_accuracy, val_class_accuracy[0], val_mean_sp, val_class_sp[0]))

if __name__ == '__main__':
    import numpy as np
    # np.random.seed(1)  # Numpy module.
    # random.seed(seed)  # Python random module.
    np.set_printoptions(threshold=9999999)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    auto_val(net)