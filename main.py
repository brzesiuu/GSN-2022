from dataset import FreiPoseConfig, FreiPoseDataset
from models import UNet


def main():
    config = FreiPoseConfig(folder_path='C:\\Users\\filip\\Downloads\\FreiHAND_pub_v2')
    dataset = FreiPoseDataset(config)
    image, heatmaps = dataset[1]
    net = UNet(16)
    xd = net(image.unsqueeze(0))


if __name__ == '__main__':
    main()
