from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tf_keras as keras

from PIL import Image
from tqdm import tqdm
import numpy as np
import copy
import cv2

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print("2025/12/13 - 13:31")

PATH = '/home/ubuntu/fresh_start/Project/'
DIR_FAKE = PATH + "big_data_step1/fake/"
DIR_REAL = PATH + "big_data_step1/real/"
MODEL    = PATH + 'model/xception_faceforensics.h5'
N_IMGS = 4000 # number of images to load /!\ MEMORY
MULTI_THRESHOLD = False # try multiple thresholds on the final predictions to maybe get a better accuracy ?
ZOOM_FACTOR = 1.0 # how much to zoom in in each picture (zooming in doesn't provide better results)
LAPLACIAN_MIN = .05 # Minimum laplacian quality for an image to be loaded

class FixedDepthwiseConv2D(keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        # Nobuco added 'groups', we need to pop them out
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(**kwargs)

print(f"Loading {MODEL} with custom patch...")

model = keras.models.load_model(
    MODEL,
    custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D}
)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

print("Model loaded successfully")
print(f"Input Shape: {model.input_shape}")


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def preprocess_image(img):
    if img is None: return None
    
    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img, (299, 299))
    
    
    # [-1, 1]
    img = img.astype(np.float32) / 127.0 - 1.0
    
    # Batch dim -> (1, 299, 299, 3)
    img_tensor = tf.convert_to_tensor(img)
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    
    return img_tensor

def get_laplacian_variance_score(img):
    if img is None:
        return 0.0

    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

   
    threshold = 500 
    score = 1 / (1 + np.exp(-(laplacian_var - threshold) / 100))

    return score
    # return 1.0 # 


def zoom_center(img, zoom_factor=1.5):
    y_size, x_size, _ = img.shape

    crop_h = int(y_size / zoom_factor)
    crop_w = int(x_size / zoom_factor)
    
    y1 = int((y_size - crop_h) / 2)
    x1 = int((x_size - crop_w) / 2)
    y2 = y1 + crop_h
    x2 = x1 + crop_w
    
    # Crop
    img_cropped = img[y1:y2, x1:x2]

    img_zoomed = cv2.resize(img_cropped, (x_size, y_size), interpolation=cv2.INTER_LINEAR)
    
    return img_zoomed

def display_image(img):
    res = img[0]
    res = res + 1
    res = res / 2.0
    return res

# real_imgs = []
fake_imgs = []

# real_imgs_names = []
fake_imgs_names = []

"""
print("loading real imgs")
iter=0
for fname in tqdm(fnames_real[:N_IMGS], desc="Processing Real Images"):
    iter += 1
    path = os.path.join(DIR_REAL, fname)
    if 'jpg' not in path: continue
    img = cv2.imread(path)
    img = np.array(img)
    if img is not None and img.shape != ():
        img = zoom_center(img, ZOOM_FACTOR)
        if get_laplacian_variance_score(img) > LAPLACIAN_MIN:
            real_imgs.append(img)
        real_imgs_names.append(fname)
"""

print("loading fake imgs")
iter=0 
for fname in tqdm(os.listdir(DIR_FAKE)[:N_IMGS], desc="Processing Fake Images"):
    iter += 1
    path = os.path.join(DIR_FAKE, fname)
    if 'jpg' not in path: continue
    img = cv2.imread(path)
    img = np.array(img)
    if img is not None and img.shape != ():
        img = zoom_center(img, ZOOM_FACTOR)
        if get_laplacian_variance_score(img) > LAPLACIAN_MIN:
            fake_imgs.append(img)
    fake_imgs_names.append(fname)

print(f"\nSuccessfully loaded {len(fake_imgs)} fake images")
min_len = min(len(fake_imgs), N_IMGS)
print("Keeping the", min_len, "first images of each dataset")

fake_imgs = fake_imgs[:min_len]

if any(img is None for img in fake_imgs):
    print("Error, None detected. One or more images failed to load.")



def deepfool(image, model, num_classes=2, overshoot=0.02, max_iter=50, shape=(299, 299, 3)):
    image_array = np.array(image)

    image_norm = tf.cast(image_array, tf.float32)
    image_norm = np.reshape(image_norm, shape)
    image_norm = image_norm[tf.newaxis, ...]

    f_image = model(image_norm).numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]
    I = I[0:num_classes]
    orig_label = I[0]
    
    input_shape = np.shape(image_norm)
    pert_image = copy.deepcopy(image_norm)
    r_tot = np.zeros(input_shape)

    loop_i = 0
    x = tf.Variable(pert_image)
    fs = model(x)
    k_i = np.argmax(np.array(fs).flatten())

    # logique classique DeepFool, source:  https://github.com/MyRespect/AdversarialAttack/blob/master/deepfool/deepfool_tf.py
    label = orig_label
    while k_i == label and loop_i < max_iter:
        pert = np.inf
        w = np.zeros(input_shape)
        
        with tf.GradientTape() as tape:
            tape.watch(x)
            fs = model(x)
            loss_value = fs[0, I[0]]
        grad_orig = tape.gradient(loss_value, x)

        for k in range(1, num_classes):
            with tf.GradientTape() as tape:
                tape.watch(x)
                fs = model(x)
                loss_value = fs[0, I[k]]
            
            cur_grad = tape.gradient(loss_value, x)
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).numpy()

            pert_k = abs(f_k) / (np.linalg.norm(tf.reshape(w_k, [-1])) + 1e-8)

            if pert_k < pert:
                pert = pert_k
                w = w_k

        r_i = (pert + 1e-4) * w / (np.linalg.norm(w) + 1e-8)
        r_tot = np.float32(r_tot + r_i)

        pert_image = image_norm + (1 + overshoot) * r_tot
        pert_image = np.clip(pert_image, -1, 1)

        x = tf.Variable(pert_image)
        fs = model(x)
        k_i = np.argmax(np.array(fs).flatten())

        loop_i += 1

    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, orig_label, k_i, pert_image

def proj_lp(v, xi, p):

    # Project on the lp ball centered at 0 and of radius xi

    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten())) # pas de 1 dans les parentheses de flatten
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v

def universal_perturbation(dataset, model, delta=0.2, max_iter_uni = np.inf, xi=10, p=np.inf, num_classes=2, overshoot=0.02, max_iter_df=10, shape=(299, 299, 3)):

    v = np.zeros((1, shape[0], shape[1], shape[2]), dtype=np.float32) # au lieu de v=0 
    fooling_rate = 0.0
    num_images =  np.shape(dataset)[0] # The images should be stacked ALONG FIRST DIMENSION

    itr = 0
    while fooling_rate < 1-delta and itr < max_iter_uni:
        # Shuffle the dataset
        np.random.shuffle(dataset)

        # Go through the data set and compute the perturbation increments sequentially
        iter = 0
        for k in tqdm(range(0, num_images), desc=f'pass #{itr}'):
            cur_img = dataset[k:(k+1), :, :, :]

            if int(np.argmax(np.array(model(cur_img)).flatten())) == int(np.argmax(np.array(model(cur_img+v)).flatten())):

                # Compute adversarial perturbation

                # Compute adversarial perturbation
                # us:   r_tot, loop_i, label, k_i, pert_image
                # here: r_tot, loop_i,        k_i, pert_image
                dr,iter,_,_,_ = deepfool(cur_img + v, model, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df, shape=shape)

                iter += 1
                # Make sure it converged...
                if iter < 10-1:
                    v = v + dr

                    # Project on l_p ball
                    v = proj_lp(v, xi, p)

        itr = itr + 1

        # Perturb the dataset with computed perturbation
        dataset_perturbed = dataset + v

        est_labels_orig = np.zeros((num_images))
        est_labels_pert = np.zeros((num_images))

        batch_size = 100
        num_batches = int(np.ceil(1.0*num_images / batch_size))

        # Compute the estimated labels in batches
        for ii in range(0, num_batches):
            m = (ii * batch_size)
            M = min((ii+1)*batch_size, num_images)
            
            pred_orig = model.predict(dataset[m:M, :, :, :], batch_size=1, verbose=0)
            max_orig = np.argmax(pred_orig, axis=1)
            est_labels_orig[m:M] = max_orig.flatten()
            
            pred_pert = model.predict(dataset_perturbed[m:M, :, :, :], batch_size=1, verbose=0)
            max_pert = np.argmax(pred_pert, axis=1)
            est_labels_pert[m:M] = max_pert.flatten()

        # Compute the fooling rate
        fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_images))
        print('FOOLING RATE =', fooling_rate)

    return v, fooling_rate

X = []
y = []
for i in fake_imgs:
    X.append(preprocess_image(i)[0])
    y.append([[0, 1]])

# for i in real_imgs:
#     X.append(preprocess_image(i)[0])
#     y.append([[1, 0]])

X = np.array(X)
y = np.array(y)

DELTA=.01
OVERSHOOT=.3

v, fr = universal_perturbation(X, model, delta=DELTA, overshoot=OVERSHOOT)

print("="*100)
print(v)
print("="*100)
print(f"Ran the adversarial perturbation, got this fooling rate: {fr*100:.2f}%")
fn = f"uap-df_delta={DELTA}_overshoot={OVERSHOOT}"
image = Image.fromarray((v[0] * 255.0).astype(np.uint8))
image.save(PATH + f'adversarial-attacks-deepfake_discovery/UAPs/{fn}.png')
np.save(PATH + f'adversarial-attacks-deepfake_discovery/UAPs/{fn}.npy', v)
image.save('out.png')
image.save('out.jpg')

print("saved as:", PATH + f'adversarial-attacks-deepfake_discovery/UAPs/{fn}.png')





