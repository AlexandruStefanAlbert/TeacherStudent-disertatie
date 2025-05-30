import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from keras.applications.densenet import layers
from tensorflow.python.estimator import keras
import tensorflow as tf

matplotlib.use('TkAgg')

# Paths
base_path = r"E:/Desktop/Disertatie_new/Disertatie/Student-Teacher Application/data/train/labeled/"
val_base_path = r"E:/Desktop/Disertatie_new/Disertatie/Student-Teacher Application/data/val/"
test_base_path = r"E:/Desktop/Disertatie_new/Disertatie/Student-Teacher Application/data/val_old/"
test_image_path = os.path.join(test_base_path)
image_path = os.path.join(base_path, 'images/')
val_image_path = os.path.join(val_base_path, 'images/')
val_mask_path = os.path.join(val_base_path, 'masks/')
mask_path = os.path.join(base_path, 'masks/')

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 20
AUTOTUNE = tf.data.AUTOTUNE
OUTPUT_CLASSES = 1
MODEL_PATH = 'teacher_mask.pth'
MODEL_PATH2 = 'teacher_mask2.pth'
NUM_IMAGES = len(os.listdir(image_path))  # verific daca fiecare imagine are o masca corespondenta
STEPS_PER_EPOCH = NUM_IMAGES // BATCH_SIZE

# GPU Check
print("Numarul de GPU-uri disponibile: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU-ul este disponibil!")
    for gpu in gpus:
        print(f"GPU detectat: {gpu}")
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("GPU nu este disponibil, se foloseste CPU.")


# Load the images and masks
def load_image(image_file, mask_file):
    # Load image
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    # Load mask
    mask = tf.io.read_file(mask_file)
    mask = tf.image.decode_jpeg(mask, channels=1)
    mask = tf.image.resize(mask, IMG_SIZE, method='nearest')  # No interpolation for masks
    mask = tf.cast(mask, tf.float32) / 255.0  # Normalize mask to [0,1]

    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")

    return image, mask


def load_test_image(image_file):
    # Load image
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    print(f"Image shape: {image.shape}")

    return image


def load_dataset(image_dir, mask_dir):
    image_files = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
    mask_files = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])

    dataset = tf.data.Dataset.from_tensor_slices((image_files, mask_files))
    dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE)
    return dataset


def load_test_dataset(image_dir):
    image_files = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])

    dataset = tf.data.Dataset.from_tensor_slices(image_files)
    dataset = dataset.map(load_test_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(8)  # sau alt batch_size potrivit
    return dataset


train_dataset = load_dataset(image_path, mask_path)
val_dataset = load_dataset(val_image_path, val_mask_path)
test_dataset = load_test_dataset(test_image_path)


# train_dataset = train_dataset.take(100)


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42, max_delta=1.0):
        super().__init__()
        self.max_delta = max_delta

    def call(self, images, labels):
        images = self.random_blur(images)  # Aplică doar blur pe imagini
        return images, labels

    def random_blur(self, image, max_delta=1.0):
        # Aplicare un blur pe imagini
        # image = tfa.image.gaussian_filter2d(image, filter_shape=(5, 5), sigma=0.8)
        image = tf.image.random_brightness(image, max_delta=max_delta)  # Modificare aleatorie a luminozității
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # Ajustează contrastul imaginii
        # Filtrare Gauss pentru reducere de zgomot
        # image = tfa.image.gaussian_filter2d(image, filter_shape=(5, 5), sigma=0.8)
        image = tf.image.resize(image, (224, 224))  # Resize to (224x224)
        return image


# Aplicare augmentare inainte de batching
augment = Augment()

train_batches = (
    train_dataset
    .cache()
    .shuffle(500)
    .batch(BATCH_SIZE)
    .map(augment, num_parallel_calls=AUTOTUNE)  # Aplică augmentarea personalizată
    .prefetch(buffer_size=AUTOTUNE)
    .repeat()  # Aceasta va repeta dataset-ul indefinit
)

val_batches = (
    val_dataset
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=AUTOTUNE)
)

test_batches = (
    test_dataset
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=AUTOTUNE)
)


class SegmentationModel:
    def __init__(self, img_size=[224, 224, 3], output_channels=1, learning_rate=3e-4):
        self.img_size = img_size
        self.output_channels = output_channels
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=self.combined_loss,
            metrics=[self.dice_coef, 'accuracy']
        )

    def _build_model(self):
        # Encoder: MobileNetV2 pretrained
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=[224, 224, 3], include_top=False
        )

        # Straturile de extagere a hartii de caracteristici
        layer_names = [
            'block_1_expand_relu',  # 64x64
            'block_3_expand_relu',  # 32x32
            'block_6_expand_relu',  # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',  # 4x4
        ]
        base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
        # Define the down-stack model (encoder)
        down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
        down_stack.trainable = False

        # Decoder: upsampling blocks
        # Definire up-stack cu droput (decoder)
        up_stack = [
            self._upsample(512, 3, apply_dropout=True),  # 4x4 -> 8x8
            self._upsample(256, 3, apply_dropout=True),  # 8x8 -> 16x16
            self._upsample(128, 3),  # 16x16 -> 32x32
            self._upsample(64, 3, apply_dropout=True),  # 32x32 -> 64x64
            self._upsample(32, 3),  # 64x64 -> 128x128
            self._upsample(16, 3),  # 128x128 -> 224x224
        ]

        inputs = layers.Input(shape=self.img_size)
        skips = down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])

        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = layers.Concatenate()([x, skip])

        last = layers.Conv2DTranspose(
            filters=self.output_channels, kernel_size=3, strides=2,
            padding='same',
            activation='sigmoid'  # pentru output [0,1]
        )
        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def _upsample(self, filters, size, apply_dropout=False):
        initializer = tf.keras.initializers.he_normal()
        result = tf.keras.Sequential()
        result.add(layers.Conv2DTranspose(filters, size, strides=2,
                                          padding='same',
                                          kernel_initializer=initializer,
                                          use_bias=False))
        result.add(layers.BatchNormalization())
        if apply_dropout:
            result.add(layers.Dropout(0.5))
        result.add(layers.ReLU())
        return result

    def train(self, train_data, val_data, epochs=20, steps_per_epoch=None):
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch
        )
        return history

    # Metoda pentru predictie
    def predict(self, images):
        return self.model.predict(images)

    def dice_coef(self, y_true, y_pred, smooth=1):
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    def dice_loss(self, y_true, y_pred, smooth=1e-6):
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        return 1 - (2. * intersection + smooth) / (union + smooth)

    def combined_loss(self, y_true, y_pred):
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)
        dice = self.dice_loss(y_true, y_pred)
        return bce + dice

    # Apicarea mastii prezise peste imaginea originala
    def create_mask(self, pred_mask):
        pred_mask = tf.math.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask[0]

    def display(self, display_list):
        plt.figure(figsize=(15, 15))
        title = ['Input Image', 'True Mask', 'Predicted Mask']

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i + 1)
            plt.title(title[i])
            plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
            plt.axis('off')
        plt.show()

    def show_predictions(self, dataset=None, num=1):
        if dataset:
            for batch in dataset.take(num):  # luăm un singur batch
                images, masks = batch
                for i in range(num):
                    image = images[i]
                    mask = masks[i]
                    pred_mask = model.predict(tf.expand_dims(image, axis=0))
                    self.display([image, mask, self.create_mask(pred_mask)])
        else:
            sample_image, sample_mask = next(iter(train_batches))
            pred_mask = model.predict(tf.expand_dims(sample_image, axis=0))
            self.display([sample_image, sample_mask, self.create_mask(pred_mask)])

    # Evaluarea modelului cu metricile IoU
    def calculate_iou(self, preds, masks):
        print(f"Predictions shape: {preds.shape}")
        print(f"Mask shape before IOU: {masks.shape}")

        intersection = np.sum(preds * masks)
        union = np.sum(preds) + np.sum(masks) - intersection
        iou = intersection / union
        return iou

    # Vizualizare predictii
    def display(self, display_list):
        plt.figure(figsize=(15, 5))
        title = ['Input Image', 'True Mask', 'Predicted Mask']

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i + 1)
            plt.title(title[i])
            plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
            plt.axis('off')
        plt.show()
        
    def show_full_results_test(self, image, pred_mask, alpha=0.5):
        """
        Functie pentru afisarea mastilor prezise pe un set de testare/validare
        Afisare:
        - imaginea originala
        - masca prezisa
        - imaginea cu overlay peste original

        Args:
            image (Tensor sau array) - imaginea originala
            pred_mask (Tensor sau array) - masca prezisă (0 sau 1)
            alpha (float) - transparența mastii prezise în overlay
        """
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # 1. Imaginea originala
        axes[0].imshow(tf.keras.utils.array_to_img(image))
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        print(pred_mask.shape)  # Verifică forma predicției

        # 2. Masca prezisa
        if len(pred_mask.shape) == 1:  # Verificare daca predictia are forma (224,)
            pred_mask = np.expand_dims(pred_mask, axis=-1)  # Adaugare canal de culoare
            pred_mask = np.tile(pred_mask, (1, 224))  # Redimensionare la (224, 224)
        if len(pred_mask.shape) == 2:  # Dacă predicția este 2D
            pred_mask = np.expand_dims(pred_mask, axis=-1)  # Adaugare canal de culoare

        axes[1].imshow(pred_mask, cmap='gray')
        axes[1].set_title('Predicted Mask')
        axes[1].axis('off')

        # 3. Suprapunere masca prezisa peste imagine originala
        axes[2].imshow(tf.keras.utils.array_to_img(image))
        axes[2].imshow(pred_mask[..., 0], cmap='Reds', alpha=alpha)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

    def save(self, path=MODEL_PATH):
        self.model.save(path)

    def load(self, path=MODEL_PATH):
        self.model = tf.keras.models.load_model(
            path,
            custom_objects={'dice_coef': self.dice_coef, 'combined_loss': self.combined_loss}
        )


model = SegmentationModel()
#model.load()

model.train(train_batches, val_batches, epochs=40, steps_per_epoch=STEPS_PER_EPOCH)
model.save(MODEL_PATH2)

def show_full_results(image, true_mask, pred_mask, alpha=0.5):
    """
    Afisare:
    - imaginea originala
    - masca reală
    - masca prezisa
    - imaginea cu overlay peste original

    Args:
        image (Tensor sau array) - imaginea originală
        true_mask (Tensor sau array) - masca reala (0 sau 1)
        pred_mask (Tensor sau array) - masca prezisa (0 sau 1)
        alpha (float) - transparența mastii prezise în overlay
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. Imaginea originala
    axes[0].imshow(tf.keras.utils.array_to_img(image))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 2. Masca adevarata
    print(true_mask.shape)
    if len(true_mask.shape) == 1:  # Verifica daca masca este de forma (224,)
        true_mask = np.expand_dims(true_mask, axis=-1)  # Adaugre canal de culoare
        true_mask = np.tile(true_mask, (1, 224))  # Redimensioneaza maca la (224, 224)
    if len(true_mask.shape) == 2:  # Verifica daca masca este 2D
        true_mask = np.expand_dims(true_mask, axis=-1)  # Adaugare canal de culoare

    axes[1].imshow(true_mask, cmap='gray')
    axes[1].set_title('True Mask')
    axes[1].axis('off')

    print(pred_mask.shape)  # Verifica forma finala a predictiei

    # 3. Obtinerea masii prezise
    if len(pred_mask.shape) == 1:  # Dacă predicția are forma (224,)
        pred_mask = np.expand_dims(pred_mask, axis=-1)  # Adaug un canal de culoare
        pred_mask = np.tile(pred_mask, (1, 224))  # Redimensioneare la (224, 224)
    if len(pred_mask.shape) == 2:  # Dacă predicția este 2D
        pred_mask = np.expand_dims(pred_mask, axis=-1)  # Adaug un canal de culoare

    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')

    # 4. Suprapunere masca prezisa peste masca originala
    axes[3].imshow(tf.keras.utils.array_to_img(image))
    axes[3].imshow(pred_mask[..., 0], cmap='Reds', alpha=alpha)
    axes[3].set_title('Overlay')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()

def Modeltest():
    #test_dataset : setul de date pentru validare/trstare
    for images in test_dataset:  # train_batches.take(3):

        images = tf.reshape(images, (-1, 224, 224, 3))
        # masks = tf.reshape(masks, (-1, 224, 224, 1)) 
        # print(f"Images shape: {images.shape}, Masks shape: {masks.shape}")

        preds = model.predict(images)

        print(f"Preds shape: {preds.shape}")  # Predictia ar trebui sa fie de forma (BATCH_SIZE, 224, 224, 1)

        preds = (preds > 0.1).astype(np.float32)

        for i in range(8):
            # iou = calculate_iou(preds[i], masks[i])
            # print(f"IoU for image {i}: {iou}")
            model.show_full_results_test(images[i], preds[i])

Modeltest()
