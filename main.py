# This is a sample Python script.
from keras.saving.save import load_model
from CNNmodels.segmentationModel import *

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

#MY_MODEL_PATH = "E:/Desktop/Disertatie_new/Disertatie/Teacher-Student Application/CNNmodels/teacher_mask.pth"
MY_MODEL_PATH = r'E:\Desktop\Disertatie_new\Disertatie\Student-Teacher Application\CNNmodels\teacher_mask.pth'

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print("MAIN")
    model = load_model(MY_MODEL_PATH, custom_objects={
        'dice_coef': dice_coef,
        'combined_loss': combined_loss
    })
    model.summary()
    # Test on a few images
    for images in test_batches:

        images = tf.reshape(images, (-1, 224, 224, 3))
        #masks = tf.reshape(masks, (-1, 224, 224, 1))
        #print(f"Images shape: {images.shape}, Masks shape: {masks.shape}")
        preds = model.predict(images)

        print(f"Preds shape: {preds.shape}")

        preds = (preds > 0.1).astype(np.float32)

        batch_size = images.shape[0]
        for i in range(batch_size):
            #iou = calculate_iou(preds[i], masks[i])
            #print(f"IoU for image {i}: {iou}")
            show_full_results_test(images[i], preds[i])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
