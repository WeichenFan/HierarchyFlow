from .base_dataset import *


def get_imgs_from_dir(path_a):
    file_a = open(path_a)
    list_a = []
    for lines in file_a.readlines():
        line = lines.strip('\n')
        list_a.append(line)
    file_a.close()

    return list_a

class DeDataset(BaseDataset):
    def __init__(self, rootA, rootB, transform, use_mc=False):

        self.Source = []

        self.transform = transform
        self.use_mc = use_mc

        imgs_A_1 = get_imgs_from_dir(rootA)
        # for i in range(len(imgs_A_1)):
        #     imgs_A_1[i] = os.path.join(rootA,imgs_A_1[i])
        #     imgs_B_1[i] = os.path.join(rootB,imgs_B_1[i])
        self.Source += imgs_A_1

        random.shuffle(imgs_A_1)


        self.initialized_ceph = False
        self.initialized_mc = False

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
        if not 's3' in self.Source[idx]:
        # if self.use_mc:
            self._init_memcached()
            source = self.read_from_mc(self.Source[idx])
            #target = self.read_from_mc(self.Target[idx])
        else:
            self._init_ceph()
            source = self.read_from_ceph(self.Source[idx])
            #target = self.read_from_ceph(self.Target[idx])
        #source_alt = self.read_from_mc(self.Source_alt[idx])
        #target_alt = self.read_from_mc(self.Target_alt[idx])
        
        source,_ = self.transform([source,source])

        return source

    def __len__(self):
        return len(self.Source)
    