##########################################################
############### Dataset management class #################
##########################################################

from keras.preprocessing.image import ImageDataGenerator


class DataManager(object):
    """docstring for DataManager"""

    def __init__(self, img_width, img_height):
        super(DataManager, self).__init__()
        self.img_width = img_width
        self.img_height = img_height

    def get_train_data(self, train_data_dir, validation_data_dir, train_batch_size, val_batch_size):
        # used to rescale the pixel values from [0, 255] to [0, 1] interval
        datagen = ImageDataGenerator(rescale=1. / 255)
        # Data augmentation for improving the model
        train_datagen_augmented = ImageDataGenerator(
            rescale=1. / 255,        # normalize pixel values to [0,1]
            #shear_range=0.2,       # randomly applies shearing transformation
            #zoom_range=0.2,        # randomly applies shearing transformation
            horizontal_flip=True,
            vertical_flip=True)  # randomly flip the images

        train_datagen_augmented2 = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        # automagically retrieve images and their classes for train and
        # validation and test sets
        print("Train data:")
        train_generator_augmented = train_datagen_augmented.flow_from_directory(
            train_data_dir,
            target_size=(self.img_width, self.img_height),
            classes=['cover', 'stego'],
            batch_size=train_batch_size,
            color_mode='grayscale',
            # save_to_dir='/home/rabii/Desktop/thesis/projects/project1/datasets/augmented_data',
            # class_mode='binary',
            shuffle=True)

        print("Validation data:")
        validation_generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(self.img_width, self.img_height),
            classes=['cover', 'stego'],
            batch_size=val_batch_size,
            color_mode='grayscale',
            # class_mode='binary',
            shuffle=True)

        return train_generator_augmented, validation_generator

    # Get only the test data
    def get_test_data(self, test_data_dir):
        # used to rescale the pixel values from [0, 255] to [0, 1] interval
        datagen = ImageDataGenerator(rescale=1. / 255)
        print("Test data:")
        test_generator = datagen.flow_from_directory(
            test_data_dir,
            target_size=(self.img_width, self.img_height),
            classes=['cover', 'stego'],
            batch_size=16,
            color_mode='grayscale',
            # class_mode='binary',
            shuffle=False)
        return test_generator
