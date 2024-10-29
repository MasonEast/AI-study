from PIL import Image
import os
import numpy as np

def preprocess_images(input_dir, output_dir, target_size=(28, 28)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        with Image.open(img_path) as img:
            # 调整尺寸
            img = img.resize(target_size)

            # 转换为数组并归一化
            img_array = np.array(img) / 255.0

            # 转换回图像以保存到输出文件夹
            processed_img = Image.fromarray((img_array * 255).astype(np.uint8))
            processed_img.save(os.path.join(output_dir, img_name))

preprocess_images(os.path.join(os.getcwd(),'img_b'), os.path.join(os.getcwd(),'img'))