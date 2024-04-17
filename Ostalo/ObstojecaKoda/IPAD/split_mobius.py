import os
import shutil

ROOT = '../MOBIUS Segmentation/MOBIUS'
MASKS_DIR = os.path.join(ROOT, 'Masks')

TRAIN_DIR = os.path.join(ROOT, 'train')
TRAIN_IMAGES_DIR = os.path.join(TRAIN_DIR, 'images')
TRAIN_LABELS_DIR = os.path.join(TRAIN_DIR, 'labels')

VAL_DIR = os.path.join(ROOT, 'val')
VAL_IMAGES_DIR = os.path.join(VAL_DIR, 'images')
VAL_LABELS_DIR = os.path.join(VAL_DIR, 'labels')

TEST_DIR = os.path.join(ROOT, 'test')
TEST_IMAGES_DIR = os.path.join(TEST_DIR, 'images')
TEST_LABELS_DIR = os.path.join(TEST_DIR, 'labels')


if __name__ == '__main__':
    # TRAIN
    # get all file names for train
    train_filenames = []
    for file in os.listdir(TRAIN_IMAGES_DIR):
        # print(file)
        if file.endswith(".jpg"):
            train_filenames.append(file.strip(".jpg"))

    print('found {0} train images'.format(len(train_filenames)))
    error_count = 0
    for filename in train_filenames:
        label_source_path = os.path.join(MASKS_DIR, filename + '_iris.png')
        try:
            shutil.copy2(label_source_path, TRAIN_LABELS_DIR)
        except:
            print('Error with file ' + str(label_source_path))
            error_count += 1

        label_source_path = os.path.join(MASKS_DIR, filename + '_pupil.png')
        try:
            shutil.copy2(label_source_path, TRAIN_LABELS_DIR)
        except:
            print('Error with file ' + str(label_source_path))
            error_count += 1

        label_source_path = os.path.join(MASKS_DIR, filename + '_sclera.png')
        try:
            shutil.copy2(label_source_path, TRAIN_LABELS_DIR)
        except:
            print('Error with file ' + str(label_source_path))
            error_count += 1

    print('error count: ' + str(error_count))

    # VAL
    val_filenames = []
    for file in os.listdir(VAL_IMAGES_DIR):
        # print(file)
        if file.endswith(".jpg"):
            val_filenames.append(file.strip(".jpg"))

    print('found {0} val images'.format(len(val_filenames)))
    error_count = 0

    for filename in val_filenames:
        label_source_path = os.path.join(MASKS_DIR, filename + '_iris.png')
        try:
            shutil.copy2(label_source_path, VAL_LABELS_DIR)
        except:
            print('Error with file ' + str(label_source_path))
            error_count += 1

        label_source_path = os.path.join(MASKS_DIR, filename + '_pupil.png')
        try:
            shutil.copy2(label_source_path, VAL_LABELS_DIR)
        except:
            print('Error with file ' + str(label_source_path))
            error_count += 1

        label_source_path = os.path.join(MASKS_DIR, filename + '_sclera.png')
        try:
            shutil.copy2(label_source_path, VAL_LABELS_DIR)
        except:
            print('Error with file ' + str(label_source_path))
            error_count += 1

    print('error count: ' + str(error_count))

    # TEST
    test_filenames = []
    for file in os.listdir(TEST_IMAGES_DIR):
        # print(file)
        if file.endswith(".jpg"):
            test_filenames.append(file.strip(".jpg"))

    print('found {0} test images'.format(len(test_filenames)))
    error_count = 0
    for filename in test_filenames:
        label_source_path = os.path.join(MASKS_DIR, filename + '_iris.png')
        try:
            shutil.copy2(label_source_path, TEST_LABELS_DIR)
        except:
            print('Error with file ' + str(label_source_path))
            error_count += 1

        label_source_path = os.path.join(MASKS_DIR, filename + '_pupil.png')
        try:
            shutil.copy2(label_source_path, TEST_LABELS_DIR)
        except:
            print('Error with file ' + str(label_source_path))
            error_count += 1

        label_source_path = os.path.join(MASKS_DIR, filename + '_sclera.png')
        try:
            shutil.copy2(label_source_path, TEST_LABELS_DIR)
        except:
            print('Error with file ' + str(label_source_path))
            error_count += 1

    print('error count: ' + str(error_count))


