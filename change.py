import numpy as np
from json import load
import cv2
from pathlib import Path


def json_to_mask(json_path, mask_path):
    """
    标注的json数据文件转换为图像的掩码
    """
    h, w = 640, 960
    mask = np.zeros([h, w, 1], np.uint8)
    with open(json_path, "r", encoding='utf-8') as f:
        json_labels = load(f)
        json_shapes = json_labels["shapes"]
        if json_shapes:
            for shape in json_shapes:
                points = shape["points"]
                # 填充
                points_array = np.array(points, dtype=np.int32)
                mask = cv2.fillPoly(mask, [points_array], 255)
                
            cv2.imencode('.png', mask)[1].tofile(
                str(mask_path)
            )
        else:
            image_name = Path(json_path).stem
            print(image_name)


# 硬编码版本的主函数
if __name__ == "__main__":
    # 硬编码指定JSON文件路径和掩码保存路径
    json_path = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/label_good/labels_my-project-name_2025-10-16-12-12-05.json"
    mask_path = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/label_good/mask.png"
    
    # 直接调用转换函数
    json_to_mask(json_path, mask_path)
    print(f"已将 {json_path} 转换为掩码图像 {mask_path}")