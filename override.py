import os 
import cv2

# Some images were corrupted. Therefore, I load them and 
# applied interpolation via opencv, then save them with same name. 

if __name__ == '__main__':
    images_dir = "trodo-v01/images"
    image_path_list = sorted(os.listdir(images_dir))

    for i in range(len(image_path_list)):
        image_path = os.path.join(images_dir, image_path_list[i])
        img = cv2.imread(image_path)
        cv2.imwrite(image_path, img)
