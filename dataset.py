import paddle.vision.transforms as T
from paddle.vision.datasets import Cifar10

train_transform = T.Compose(
    [
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225], #使用imagenet的归一化参数
            to_rgb=True,
        ),
        T.RandomVerticalFlip(),
        T.RandomHorizontalFlip(),
        T.RandomRotation(20)
    ]
)


test_transform = T.Compose(
    [
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225], #使用imagenet的归一化参数
            to_rgb=True,
        ),
    ]
)

train_dataset = Cifar10(
    mode="train",
    transform=train_transform,  # apply transform to every image
    backend="cv2",  # use OpenCV as image transform backend
)

test_dataset = Cifar10(
    mode="test",
    transform=test_transform,  # apply transform to every image
    backend="cv2",  # use OpenCV as image transform backend
)
