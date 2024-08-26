import cv2
import tensorflow as tf
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

img_height = 512
img_width = 512
img_channels = 3
saved_model = tf.keras.models.load_model('/home/harshveer.singh/python poject/drone_feed_image_segmentation_model.h5')

def visualize_results(x, y_pred, idx):
    plt.figure(figsize=(12, 4))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(x[idx])
    plt.axis('off')
    
    ## Ground truth mask
    #plt.subplot(1, 3, 2)
    #plt.title("Ground Truth Mask")
    #plt.imshow(np.squeeze(y_true[idx]), cmap='gray')
    #plt.axis('off')
   
    
    # Predicted mask
    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(np.squeeze(y_pred[idx]),cmap='gray')
    
    plt.show()

video_test_images = np.zeros((5, img_height,img_width,img_channels), dtype = np.uint8)
cap = cv2.VideoCapture('/dev/video4') #USB video devices are usualy located here. Check /dev/video for video devices
while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = resize(frame,(img_height,img_width), mode='constant',preserve_range=True)
    video_test_images[0] = image
    preds_test1 = saved_model.predict(video_test_images, verbose=1)

    visualize_results(video_test_images,preds_test1, 0)

    
    cv2.imshow('cam feed', preds_test1[0])
    

    # press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
