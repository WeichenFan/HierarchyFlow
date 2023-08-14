from .base_dataset import *


def get_imgs_from_dir(path_a,path_b):
    file_a = open(path_a)
    list_a = []
    for lines in file_a.readlines():
        line = lines.strip('\n')
        list_a.append(line)
    file_a.close()

    file_b = open(path_b)
    list_b = []
    for lines in file_b.readlines():
        line = lines.strip('\n')
        list_b.append(line)
    file_b.close()

    return list_a,list_b

class CommonDataset(BaseDataset):
    def __init__(self, rootA, rootB, transform, use_mc=False):

        self.Source = []
        self.Target = []
        self.Source_alt = []
        self.Target_alt = []
        self.transform = transform
        self.use_mc = use_mc

        imgs_A_1,imgs_B_1 = get_imgs_from_dir(rootA,rootB)
        # for i in range(len(imgs_A_1)):
        #     imgs_A_1[i] = os.path.join(rootA,imgs_A_1[i])
        #     imgs_B_1[i] = os.path.join(rootB,imgs_B_1[i])
        self.Source += imgs_A_1
        self.Target += imgs_B_1

        random.shuffle(imgs_A_1)
        random.shuffle(imgs_B_1)

        self.Source_alt += imgs_A_1
        self.Target_alt += imgs_B_1

        c = list(zip(self.Source, self.Target))
        random.shuffle(c)
        self.Source,self.Target = zip(*c)

        c = list(zip(self.Source_alt, self.Target_alt))
        random.shuffle(c)
        self.Source_alt, self.Target_alt = zip(*c)

        self.initialized_ceph = False
        self.initialized_mc = False
        print('len: ',len(self.Source))

    def read_from_mc(self,filename):
        value = mc.pyvector()
        self.mclient.Get(filename, value)
        value_str = mc.ConvertBuffer(value)
        img = self._common_loader()(value_str)
        img = np.array(img)

        import torchvision
        img = torchvision.transforms.ToPILImage()(img)
        return img

    def read_from_ceph(self,filename):
        value = self.s3_client.Get(filename)
        img = np.frombuffer(value, dtype=np.uint8)
        img = self._common_loader()(img)
        img = np.array(img)

        import torchvision
        img = torchvision.transforms.ToPILImage()(img)
        return img

    def __getitem__(self, idx):
        #if not 's3' in self.Source[idx]:
        # if self.use_mc:
        self._init_memcached()
        source = self.read_from_mc(self.Source[idx])
            #target = self.read_from_mc(self.Target[idx])
        # else:
        #     self._init_ceph()
        #     source = self.read_from_ceph(self.Source[idx])
        #     #target = self.read_from_ceph(self.Target[idx])
        #if not 's3' in self.Target[idx]:
        # if self.use_mc:
       # self._init_memcached()
        #source = self.read_from_mc(self.Source[idx])
        target = self.read_from_mc(self.Target[idx])
        # else:
        #     self._init_ceph()
        #     #source = self.read_from_ceph(self.Source[idx])
        #     target = self.read_from_ceph(self.Target[idx])
        
        #source_alt = self.read_from_mc(self.Source_alt[idx])
        #target_alt = self.read_from_mc(self.Target_alt[idx])
        
        source,target = self.transform([source,target])
        return source,target,self.Source[idx].split('/')[-1],self.Target[idx].split('/')[-1]

    def __len__(self):
        return len(self.Source)
    