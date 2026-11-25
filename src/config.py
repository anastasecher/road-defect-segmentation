SOURCE_CLASSES = {0: "pothole", 1: "crack", 2: "manhole"}

CLASS_NAMES = ["background", "pothole", "crack", "manhole"]
CLASS_NAMES_RU = ["фон", "выбоина", "трещина", "люк"]
NUM_CLASSES = len(CLASS_NAMES)
DEFECT_CLASS_IDS = [1, 2, 3]

CLASS_COLORS = {
    0: (0, 0, 0),
    1: (230, 40, 40),
    2: (255, 210, 0),
    3: (40, 120, 255),
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

DEFAULT_IMG_H = 352
DEFAULT_IMG_W = 640

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SEED = 42
