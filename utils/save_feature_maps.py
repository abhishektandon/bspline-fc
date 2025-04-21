from torch.nn import Softmax
import torch
import time
import os
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from torch.utils.data import  DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import init
from torch.nn import Conv2d, Parameter, DataParallel
from loss.loss import AdaFace

# Set seed for CPU
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

img_width = 224
img_height = 224
channels = 3
embedding_dim = 512

size1_new = int(np.floor(img_width/2))
size2_new = int(np.floor(img_height/2))

size1_new = int(np.ceil(size1_new/2))
size2_new = int(np.ceil(size2_new/2))

size1_new = int(np.ceil(size1_new/2))
size2_new = int(np.ceil(size2_new/2))

size1_new = int(np.ceil(size1_new/2))
size2_new = int(np.ceil(size2_new/2))

size1_new = int(np.floor(size1_new/2))
size2_new = int(np.floor(size2_new/2))


def get_train_dataset(imgs_folder):
    if channels != 3:
        train_transform = trans.Compose([
            trans.Grayscale(num_output_channels=channels), 
            trans.Resize((img_width, img_height)), 
            trans.RandomHorizontalFlip(),
            trans.ToTensor(),
            trans.Normalize((0.5,), (0.5,))
            
        ])
    else:
        train_transform = trans.Compose([
            trans.Resize((img_width, img_height)), 
            trans.RandomHorizontalFlip(),
            trans.ToTensor(),
            trans.Normalize((0.5,), (0.5,))
        ])
    ds = ImageFolder(imgs_folder, train_transform)
    class_num = ds[-1][1] + 1
    return ds, class_num

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output


class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

    def __repr__(self):
        return self.__class__.__name__ + '('                + 'in_features=' + str(self.in_features)                + ', out_features=' + str(self.out_features)                + ', s=' + str(self.s)                + ', m=' + str(self.m) + ')'


class SphereProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    """
    def __init__(self, in_features, out_features, m=4):
        super(SphereProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '('                + 'in_features=' + str(self.in_features)                + ', out_features=' + str(self.out_features)                + ', m=' + str(self.m) + ')'


class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim = 512):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        #out = self.gamma*out
        out = self.gamma*out + x
        return out

class ECAAttention(nn.Module):

    def __init__(self, kernel_size=5):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return x*y.expand_as(x)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class IRBlock(nn.Module): 
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.PReLU(),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNetFace(nn.Module):
    def __init__(self, block, layers, use_se=True):
        self.inplanes = 64
        self.use_se = use_se
        super(ResNetFace, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.sattention = PAM_Module()
        self.cattention = ECAAttention(kernel_size = 5)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()
        
        self.fc5 = nn.Linear(512 * (size1_new) * (size2_new), embedding_dim)
        self.bn5 = nn.BatchNorm1d(embedding_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        conv1 = x
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        layer1 = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        layer4 = x
        x = self.bn4(x)
        x = self.dropout(x)
        
        x = self.maxpool(x)
        
        
        s_x = self.sattention(x) #spatial or positional attention.
        c_x = self.cattention(x)
        
        x = c_x + s_x #Element Sum Concat of Attention Outputs.
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.bn5(x)

        if opt.metric == "adaface":
            norm = torch.norm(x, 2, 1, True)
            x = torch.div(x, norm)
            return x, norm, conv1, layer1, layer4
        
        return x

def resnet_18(use_se=True, **kwargs):
    
    model = ResNetFace(IRBlock, [2, 2, 2, 2], use_se=use_se, **kwargs)
    return model

dataset_name = 'forehead-v1-labeled'

ds, class_num = get_train_dataset(f"{dataset_name}/train")
ds2, class_num2 = get_train_dataset("forehead-v1-labeled/test")

# os.makedirs("checkpoints", exist_ok=True)

class Config(object):
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = class_num
    metric = 'adaface'
    easy_margin = False
    use_se = False
    #loss = "cross_entropy"
    loss = 'focal_loss'

    pin_memory = True
    
    display = True
    finetune = False

    save_interval = 100

    train_batch_size = 16  # batch size
    test_batch_size = 32

    input_shape = (channels, img_width, img_height)

    optimizer = 'adam'

    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    num_workers = 2  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    
    max_epoch = 100
    lr = 3e-4  # initial learning rate before 1e-3
    lr_step = 20
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4


opt = Config()


loader = DataLoader(ds, batch_size= opt.train_batch_size, shuffle=True, pin_memory= opt.pin_memory, num_workers= opt.num_workers)
test_loader = DataLoader(ds2, batch_size = opt.test_batch_size)
class_names = ds.classes

device = torch.device("cuda")
if opt.loss == 'focal_loss':
    criterion = FocalLoss(gamma=2)

else:
    criterion = torch.nn.CrossEntropyLoss()

model = resnet_18(use_se=opt.use_se)


if opt.metric == 'add_margin':
    metric_fc = AddMarginProduct(embedding_dim, opt.num_classes, s=300, m=0.35)
elif opt.metric == 'arc_margin':
    metric_fc = ArcMarginProduct(embedding_dim, opt.num_classes, s=300, m=0.35, easy_margin=opt.easy_margin)
elif opt.metric == 'sphere':
    metric_fc = SphereProduct(embedding_dim, opt.num_classes, m=4)
elif opt.metric == 'adaface':
    metric_fc = AdaFace(embedding_dim, opt.num_classes)
else:
    metric_fc = nn.Linear(embedding_dim, opt.num_classes)

model.to(device)
model = DataParallel(model)
metric_fc.to(device)
metric_fc = DataParallel(metric_fc)
if opt.optimizer == 'sgd':
    optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                lr=opt.lr, weight_decay=opt.weight_decay)
else:
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                 lr=opt.lr, weight_decay=opt.weight_decay)
scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1) #decays learning rate by gamma every step_size epochs


def save_model(model, save_path, name, iter_cnt):
    os.makedirs(save_path, exist_ok=True)
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name
    
    
def load_model(model, load_path):
    # Load the model weights from the specified path
    model.load_state_dict(torch.load(load_path))

    return model

saved_weights_path = ''
model.load_state_dict(torch.load(saved_weights_path))
print('loaded model from:', saved_weights_path)

start_time = time.time()

class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        return super(ImageFolderWithPaths, self).__getitem__(index) + (self.imgs[index][0],)


def get_dataset_modified(imgs_folder):
    if channels != 3:
        train_transform = trans.Compose([
            trans.Grayscale(num_output_channels=channels),
            trans.Resize((img_width, img_height)),
            # trans.RandomHorizontalFlip(),
            trans.ToTensor(),
            trans.Normalize((0.5,), (0.5,))
        ])
    else:
        train_transform = trans.Compose([
            trans.Resize((img_width, img_height)),
            # trans.RandomHorizontalFlip(),
            trans.ToTensor(),
            trans.Normalize((0.5,), (0.5,))
        ])
    ds = ImageFolderWithPaths(imgs_folder, train_transform)
    class_num = ds[-2][1] + 1
    return ds, class_num

dataset_name = 'forehead-v1-labeled'
actual_path = f'{dataset_name}/train/' 

ds3, cnum = get_dataset_modified(actual_path)
print("dataset:", dataset_name)

new_loader = DataLoader(ds3, batch_size = 1)

vis_path = f'visualization/{dataset_name}'

os.makedirs(vis_path, exist_ok=True)

embedding_dict_test = {}
embedding_dict_train = {}
with torch.no_grad():
    model.eval()
    for i, data in tqdm(enumerate(new_loader, 0)):
            images, labels, path = data
            images = images.to(device)
            feature_test, norm, conv1, layer1, layer4 = model(images) #512 dimensional embedding
            
            layers = {'layer1': layer1}

            for l in layers.items():
                layer_name, layer = l
                tensor = torch.mean(layer.squeeze(0), axis=0)
                print(tensor.shape)
                spl = path[0].split('/')
                id = spl[-2]
                pose = spl[-1]
                os.makedirs(os.path.join(vis_path, layer_name+'/'+id), exist_ok=True)
                save_path = f'{vis_path}/{layer_name}/{str(id)}/{pose}'
                activation_map = tensor.cpu().numpy()
                activation_map = (activation_map - np.min(activation_map)) / (np.max(activation_map) - np.min(activation_map))

                activation_map = activation_map[activation_map > 0.5]
                activation_sum = np.sum(activation_map)

                pil_img = trans.ToPILImage()(tensor.cpu())
                plt.imshow(pil_img.resize((305, 112)), cmap='jet')
                plt.tight_layout()
                plt.axis('off')
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
