#coding:utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] =  '0'
import sys
import torch
import cv2
from imgaug import augmenters as iaa
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
os.environ['KMP_DUPLICATE_LIB_OK']='True'
pwd = os.path.abspath('./')
RESIZE_SIZE=112

def delTensorByDim(arr,index=0,dim=0):
    if dim==0:
        arr1 = arr[0:index,:,:,:]
        arr2 = arr[index+1:,:,:,:]
    elif dim==1:
        arr1 = arr[:,0:index,:,:]
        arr2 = arr[:,index+1:,:,:]
    return torch.cat((arr1,arr2),dim=dim)

def delFcTensorByDim(arr,index=0,dim=0):
    if dim==0:
        arr1 = arr[0:index,:]
        arr2 = arr[index+1:,:]
    elif dim==1:
        arr1 = arr[:,0:index,]
        arr2 = arr[:,index+1:]
    return torch.cat((arr1,arr2),dim=dim)

def sparsity(model):
    # Return global model sparsity
    a, b = 0., 0.
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a
# 按照L2 模选择要置零的核
# 比较置零后的准确率，待实现
# 将卷积核去掉：已实现90%
#    group组卷积核 输入通道暂不裁减，设置为0即可，group卷积核的上一次不应该按照l1 裁剪，
#    group组卷集核 输出通道正常裁减，正常影响下一层的裁减
#    fc层只在模型的最后存在，比较好处理，只用把相应的层裁减掉即可
#    bn层，pooling层，shotcut也需要处理:待处理

def prune_a_kernel_channel(model):
    print('Pruning model by kernel_channel... ')
    print(' Before: %.3g global sparsity' % sparsity(model))
    pre_layer=None
    pre_ind = None
    for name, m in model.named_modules():
        #import pdb
        #pdb.set_trace()
        if isinstance(m, torch.nn.Conv2d):
            #import pdb
            #pdb.set_trace()
            print("before prune:")
            print(m)
            print(m.weight.shape)
            print(20*"-") 
            #import pdb
            #pdb.set_trace()
            # 先把受前一层影响的卷积核去掉
            if pre_layer is not None and pre_ind is not None:
                if m.groups == 1:
                    for j in range(1,int(m.weight.data.shape[1]/2)+1):
                        m.weight.data=delTensorByDim(m.weight.data,pre_ind[-j],dim=1)
                    m.in_channels=m.weight.data.shape[1]
                else:
                    pre_layer.weight.data= data_backup
                    pre_layer.out_channels=data_backup.shape[0]
            mw2=m.weight* m.weight
            (ts,ind_l2)=mw2.sum(dim=[1,2,3]).sort()
            # 再把当前需要剪裁的卷积组去掉
            data_backup = m.weight.data.new_tensor(m.weight.data)
            oc=m.weight.shape[0]
            print("oc=%s"%(oc))
            #import pdb
            #pdb.set_trace()
            ind_prune,_=ind_l2[0:int(oc/2)].sort()
            for i in range(1,int(oc/2)+1):
                index = ind_prune[-i].item()
                #m.weight.data[index,:,:,:].zero_()
                m.weight.data=delTensorByDim(m.weight.data,index,dim=0)
            m.out_channels=m.weight.shape[0]
            

            print("after prune:")
            print(m)
            print(m.weight.shape)
            
            pre_layer=m
            pre_ind=ind_prune
        if isinstance(m, torch.nn.Linear):
            for j in range(1,int(m.weight.data.shape[1]/2)+1):
                m.weight.data=delFcTensorByDim(m.weight.data,pre_ind[-j],dim=1)
            m.in_features=m.weight.data.shape[1]
            
    import pdb
    pdb.set_trace()
    # 需要统一处理一下bn层
    # 需要统一处理一下fc层
    print(' After: %.3g global sparsity' % sparsity(model))

def prune(model, amount=0.3):
    # Prune model to requested global sparsity
    import torch.nn.utils.prune as prune
    print('Pruning model... ')
    print(' Before: %.3g global sparsity' % sparsity(model))
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            #import pdb
            #pdb.set_trace()
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            prune.remove(m, 'weight')  # make permanent
    print(' After: %.3g global sparsity' % sparsity(model))
def TTA_36_cropps(image, target_shape=(32, 32, 3)):
    image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))

    width, height, d = image.shape
    target_w, target_h, d = target_shape

    start_x = ( width - target_w) // 2
    start_y = ( height - target_h) // 2

    starts = [[start_x, start_y],

              [start_x - target_w, start_y],
              [start_x, start_y - target_w],
              [start_x + target_w, start_y],
              [start_x, start_y + target_w],

              [start_x + target_w, start_y + target_w],
              [start_x - target_w, start_y - target_w],
              [start_x - target_w, start_y + target_w],
              [start_x + target_w, start_y - target_w],
              ]

    images = []

    for start_index in starts:
        image_ = image.copy()
        x, y = start_index

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        if x + target_w >= RESIZE_SIZE:
            x = RESIZE_SIZE - target_w-1
        if y + target_h >= RESIZE_SIZE:
            y = RESIZE_SIZE - target_h-1

        zeros = image_[x:x + target_w, y: y+target_h, :]

        image_ = zeros.copy()

        zeros = np.fliplr(zeros)
        image_flip_lr = zeros.copy()

        zeros = np.flipud(zeros)
        image_flip_lr_up = zeros.copy()

        zeros = np.fliplr(zeros)
        image_flip_up = zeros.copy()

        images.append(image_.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip_lr.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip_up.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip_lr_up.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))

    return images
class FaceAnti:
    def __init__(self):
        from FaceBagNet_model_A import Net
        self.net = Net(num_class=2,is_first_bn=True)
        #model_path = os.path.join(pwd, 'models', 'model_A_color_64', 'checkpoint', 'global_min_acer_model.pth')
        model_path = "./global_min_acer_model.pth"
        if torch.cuda.is_available():
            state_dict = torch.load(model_path, map_location='cuda')
        else:
            state_dict = torch.load(model_path, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)
        #self.net.load_state_dict(state_dict)
        self.net.eval()
        if torch.cuda.is_available():
            self.net = self.net.cuda()
    def classify(self,color):
        return self.detect(color)
    def detect(self,color):
        #color = cv2.imread(imgpath,1)
        color = cv2.resize(color,(RESIZE_SIZE,RESIZE_SIZE))
        
        def color_augumentor(image, target_shape=(64, 64, 3), is_infer=False):
            if is_infer:
                augment_img = iaa.Sequential([
                    iaa.Fliplr(0),
                ])
            image =  augment_img.augment_image(image)
            image = TTA_36_cropps(image, target_shape)
            return image

        color = color_augumentor(color, target_shape=(64, 64, 3), is_infer=True)

        n = len(color)
        color = np.concatenate(color, axis=0)

        image = color
        image = np.transpose(image, (0, 3, 1, 2))
        image = image.astype(np.float32)
        image = image / 255.0
        input_image = torch.FloatTensor(image)
        if (len(input_image.size())==4) and torch.cuda.is_available():
            input_image = input_image.unsqueeze(0).cuda()
        elif (len(input_image.size())==4) and not torch.cuda.is_available():
            input_image = input_image.unsqueeze(0)
        
        b, n, c, w, h = input_image.size()
        input_image = input_image.view(b*n, c, w, h)
        if torch.cuda.is_available():
            input_image = input_image.cuda()

    
        with torch.no_grad():
            logit,_,_   = self.net(input_image)
            logit = logit.view(b,n,2)
            logit = torch.mean(logit, dim = 1, keepdim = False)
            prob = F.softmax(logit, 1)

        print('probabilistic：', prob)
        print('predict: ', np.argmax(prob.detach().cpu().numpy()))
        return np.argmax(prob.detach().cpu().numpy())
     
#Val/0000/000037-color.jpg Val/0000/000037-depth.jpg Val/0000/000037-ir.jpg 1
#Val/0000/000038-color.jpg Val/0000/000038-depth.jpg Val/0000/000038-ir.jpg 0
#Val/0000/000041-color.jpg Val/0000/000041-depth.jpg Val/0000/000041-ir.jpg 0
#Val/0000/000042-color.jpg Val/0000/000042-depth.jpg Val/0000/000042-ir.jpg 1
#Val/0000/000044-color.jpg Val/0000/000044-depth.jpg Val/0000/000044-ir.jpg 0

if __name__=="__main__":        
        
    FA = FaceAnti()
    import pdb
    pdb.set_trace()
    img = cv2.imread('1.jpg',1)
    FA.detect(img)
    #prune(FA.net)
    prune_a_kernel_channel(FA.net)
    sparsity(FA.net)
    FA.detect(img)
    # Return global model sparsity
    #FA.detect('./CASIA-SURF/Val/0000/000037-color.jpg')
    #FA.detect('./CASIA-SURF/Val/0000/000038-color.jpg')
    #FA.detect('./CASIA-SURF/Val/0000/000039-color.jpg')
    #FA.detect('./CASIA-SURF/Val/0000/000040-color.jpg')
    #FA.detect('./CASIA-SURF/Val/0000/000041-color.jpg')
    #FA.detect('./CASIA-SURF/Val/0000/000042-color.jpg')
