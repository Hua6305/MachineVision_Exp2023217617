import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


class BikeDetector:
    """共享单车检测器"""

    def __init__(self):
        """初始化检测器"""
        # 使用预训练的YOLOv5模型（基于COCO数据集训练）
        # COCO数据集包含自行车类别，可以直接用于共享单车检测
        print("加载YOLOv5模型（基于COCO数据集训练）...")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, device='cpu')
        self.model.eval()
        print("模型加载完成")

        # 共享单车品牌的颜色特征
        self.bike_colors = {
            'mobike': ([0, 100, 100], [10, 255, 255]),  # 橙色
            '美团': ([20, 100, 100], [30, 255, 255]),  # 黄色
            'hellobike': ([100, 100, 100], [130, 255, 255])  # 蓝色
        }

    def detect_objects(self, image_path):
        """
        检测图像中的目标

        参数:
            image_path: 输入图像路径

        返回:
            results: 检测结果
        """
        # 读取图像
        img = Image.open(image_path)

        # 使用YOLOv5检测
        results = self.model(img)

        return results

    def filter_bikes(self, results, debug=False):
        """筛选出自行车目标"""
        bikes = []

        # 获取检测结果
        detections = results.xyxy[0].cpu().numpy()

        # 降低置信度阈值以提高检测率
        confidence_threshold = 0.25  # 从0.5降低到0.25

        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection

            # 类别1表示自行车（COCO数据集中bicycle的类别ID是1）
            if int(cls) == 1 and conf > confidence_threshold:
                # 计算边界框面积，过滤太小的检测框（可能是误检）
                area = (x2 - x1) * (y2 - y1)
                if area > 500:  # 最小面积阈值，过滤太小的框
                    bikes.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf)
                    })

        # 如果检测结果太多，按置信度排序，保留前N个
        if len(bikes) > 20:
            bikes.sort(key=lambda x: x['confidence'], reverse=True)
            bikes = bikes[:20]
            if debug:
                print(f"检测到过多目标，保留置信度最高的20个")

        return bikes

    def classify_bike_brand(self, img, bbox):
        """
        根据颜色分类共享单车品牌

        参数:
            img: 原始图像
            bbox: 边界框 [x1, y1, x2, y2]

        返回:
            brand: 单车品牌
        """
        x1, y1, x2, y2 = bbox

        # 裁剪单车区域
        bike_region = img[y1:y2, x1:x2]

        # 转换到HSV空间
        hsv = cv2.cvtColor(bike_region, cv2.COLOR_BGR2HSV)

        # 检测各品牌颜色
        max_ratio = 0
        detected_brand = 'unknown'

        for brand, (lower, upper) in self.bike_colors.items():
            # 创建颜色掩码
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

            # 计算颜色占比
            ratio = np.sum(mask > 0) / mask.size

            if ratio > max_ratio and ratio > 0.1:
                max_ratio = ratio
                detected_brand = brand

        return detected_brand

    def process_image(self, image_path, output_path='result.jpg', debug=False):
        """
        完整的检测流程

        参数:
            image_path: 输入图像路径
            output_path: 输出图像路径

        返回:
            bike_info: 检测到的单车信息列表
        """
        # 1. 读取图像
        img = cv2.imread(image_path)

        # 2. 目标检测
        if debug:
            print("开始目标检测...")
        results = self.detect_objects(image_path)

        # 3. 筛选自行车
        bikes = self.filter_bikes(results, debug=debug)
        
        if debug:
            print(f"初步检测到 {len(bikes)} 个自行车目标")

        # 4. 分类品牌
        bike_info = []
        for bike in bikes:
            bbox = bike['bbox']
            brand = self.classify_bike_brand(img, bbox)

            bike_info.append({
                'bbox': bbox,
                'confidence': bike['confidence'],
                'brand': brand
            })

            # 在图像上绘制
            x1, y1, x2, y2 = bbox

            # 选择颜色
            colors = {
                'mobike': (0, 165, 255),
                '美团': (0, 255, 255),
                'hellobike': (255, 144, 30),
                'unknown': (128, 128, 128)
            }
            color = colors.get(brand, (128, 128, 128))

            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = f'{brand}: {bike["confidence"]:.2f}'
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 5. 保存结果
        cv2.imwrite(output_path, img)
        print(f"结果图像已保存: {output_path}")

        # 6. 输出检测结果
        print("=" * 50)
        print("共享单车检测结果")
        print("=" * 50)
        print(f"任务输入: 共享单车照片 ({image_path})")
        print(f"训练集: COCO")
        print("=" * 50)
        print(f"任务输出: 共享单车位置")
        print(f"检测到 {len(bike_info)} 辆共享单车")
        print("-" * 50)
        
        for i, info in enumerate(bike_info):
            bbox = info['bbox']
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1
            
            print(f"单车 {i + 1}:")
            print(f"  位置: 边界框=({x1}, {y1}, {x2}, {y2})")
            print(f"  中心点: ({center_x}, {center_y})")
            print(f"  尺寸: {width} x {height} 像素")
            print(f"  品牌: {info['brand']}")
            print(f"  置信度: {info['confidence']:.2%}")
            print("-" * 50)
        
        print("=" * 50)

        return bike_info


def main():
    """主函数"""
    detector = BikeDetector()

    # 处理图像
    image_path = 'TempDragFile_20251227_112214.jpg'  # 替换为您的共享单车照片路径
    output_path = 'detection_result.jpg'
    
    bike_info = detector.process_image(image_path, output_path, debug=True)

    # 显示结果
    if len(bike_info) > 0:
        img = cv2.imread(output_path)
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('校园共享单车检测结果 (Campus Shared Bike Detection Result)', fontsize=14)
        plt.axis('off')
        plt.savefig('detection_result.png', dpi=300, bbox_inches='tight')
        print("可视化结果已保存: detection_result.png")
        plt.show()
    else:
        print("未检测到共享单车，请检查输入图像")


if __name__ == "__main__":
    main()