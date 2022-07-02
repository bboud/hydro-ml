import enum

class Mode(enum.Enum):
    NOISE = 1
    REAL = 2
    GENERATED = 3

    NONE = -1

class ModelType(enum.Enum):
    GENERATOR = 0
    DISCRIMINATOR = 1

    NONE = -1

class ImageChannel():
    def __init__(self):
        self.mode = Mode.NONE
        self.type = ModelType.NONE
        self.batch_images = []

    def __len__(self):
        return len(self.batch_images)

    def set_mode(self, new_mode):
        self.mode = new_mode

    def get_mode(self):
        return self.mode

    def get_type(self):
        return self.type

    def push(self, image_data, name):
        self.batch_images.append((image_data, name, self.mode))

    def pop(self):
        return self.batch_images.pop()

    def reset(self):
        self.mode = Mode.NONE
        self.batch_images.clear()
