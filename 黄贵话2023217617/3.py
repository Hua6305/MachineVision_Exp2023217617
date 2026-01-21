import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from torchvision import datasets

# 设置中文字体
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class DigitDataset(Dataset):
    """手写数字数据集"""

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class DigitCNN(nn.Module):
    """卷积神经网络模型 - 增强版"""

    def __init__(self):
        super(DigitCNN, self).__init__()

        # 卷积层 - 增加深度和通道数
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class StudentIDRecognition:
    """学号识别系统"""

    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DigitCNN().to(self.device)
        # 如果提供了模型路径，加载已保存的模型
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"已加载模型: {model_path}")
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def train(self, train_loader, val_loader=None, epochs=10):
        """训练模型"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

        print("开始训练...")
        best_acc = 0.0

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 统计
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # 每100个batch打印一次进度
                if (batch_idx + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], '
                          f'Loss: {loss.item():.4f}')

            # 更新学习率
            scheduler.step()

            # 计算训练准确率
            train_accuracy = 100. * correct / total
            avg_loss = running_loss / len(train_loader)

            # 如果有验证集，计算验证准确率
            if val_loader is not None:
                val_acc = self.validate(val_loader)
                print(f'Epoch [{epoch + 1}/{epochs}], '
                      f'Train Loss: {avg_loss:.4f}, '
                      f'Train Acc: {train_accuracy:.2f}%, '
                      f'Val Acc: {val_acc:.2f}%')
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(self.model.state_dict(), 'digit_model.pth')
                    print(f'模型已保存 (最佳验证准确率: {best_acc:.2f}%)')
            else:
                print(f'Epoch [{epoch + 1}/{epochs}], '
                      f'Loss: {avg_loss:.4f}, '
                      f'Accuracy: {train_accuracy:.2f}%')

    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        self.model.train()
        return 100. * correct / total

    def recognize_student_id(self, image_path, debug=False):
        """
        识别学号照片

        参数:
            image_path: 学号照片路径
            debug: 是否显示调试信息

        返回:
            student_id: 识别出的学号字符串
        """
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            img = Image.open(image_path)
            img = np.array(img.convert('RGB'))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 预处理：分割数字
        digits, digit_boxes = self.segment_digits(img, debug=debug)

        if len(digits) == 0:
            print("警告: 未检测到任何数字!")
            return ""

        if debug:
            print(f"检测到 {len(digits)} 个数字区域")

        # 识别每个数字
        self.model.eval()
        student_id = ""
        confidences = []
        results = []

        with torch.no_grad():
            for i, digit_img in enumerate(digits):
                # 预处理数字图像，确保与MNIST格式一致
                digit_processed = self.preprocess_digit(digit_img, digit_boxes[i] if i < len(digit_boxes) else None)

                # 转换为tensor
                digit_tensor = self.transform(digit_processed).unsqueeze(0).to(self.device)

                # 预测
                output = self.model(digit_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = probabilities.max(1)
                predicted = predicted.item()
                confidence = confidence.item()

                # 获取top-3预测结果用于调试
                top3_probs, top3_indices = torch.topk(probabilities, 3, dim=1)
                top3_probs = top3_probs[0].cpu().numpy()
                top3_indices = top3_indices[0].cpu().numpy()

                student_id += str(predicted)
                confidences.append(confidence)
                results.append({
                    'predicted': predicted,
                    'confidence': confidence,
                    'top3': list(zip(top3_indices, top3_probs)),
                    'box': digit_boxes[i] if i < len(digit_boxes) else None
                })

                if debug:
                    top3_str = ", ".join([f"{idx}({prob:.3f})" for idx, prob in zip(top3_indices, top3_probs)])
                    print(f"数字 {i + 1}: 预测={predicted}, 置信度={confidence:.3f}, Top3: [{top3_str}]")

        if debug:
            print(f"平均置信度: {np.mean(confidences):.3f}")

        return student_id

    def preprocess_digit(self, digit_img, box_info=None):
        """预处理单个数字图像，使其更接近MNIST格式
        
        关键改进：
        1. 增加笔画粗细，使细线条更接近MNIST的粗笔画
        2. 正确处理白底黑字到黑底白字的转换
        3. 居中放置并保持适当比例
        4. 对斜体数字进行矫正
        """
        # 转换为numpy数组
        if isinstance(digit_img, Image.Image):
            digit_array = np.array(digit_img.convert('L'))
        else:
            digit_array = digit_img.copy()

        # 确保是灰度图
        if len(digit_array.shape) == 3:
            digit_array = cv2.cvtColor(digit_array, cv2.COLOR_BGR2GRAY)

        # 二值化，确保数字是白色（255），背景是黑色（0），与MNIST一致
        mean_val = np.mean(digit_array)
        if mean_val > 127:  # 白底黑字，需要反转
            _, digit_array = cv2.threshold(digit_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:  # 黑底白字，直接二值化
            _, digit_array = cv2.threshold(digit_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 【关键】对所有数字进行膨胀，使笔画更粗，更接近MNIST风格
        kernel = np.ones((3, 3), np.uint8)
        digit_array = cv2.dilate(digit_array, kernel, iterations=2)

        # 找到边界框
        rows = np.any(digit_array > 0, axis=1)
        cols = np.any(digit_array > 0, axis=0)
        if not np.any(rows) or not np.any(cols):
            return Image.fromarray(np.zeros((28, 28), dtype=np.uint8))

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # 裁剪
        digit_cropped = digit_array[y_min:y_max + 1, x_min:x_max + 1]
        h, w = digit_cropped.shape

        # 【新增】检测并矫正斜体
        # 计算数字的倾斜角度
        if h > 10 and w > 5:
            # 找到所有白色像素的坐标
            white_pixels = np.where(digit_cropped > 0)
            if len(white_pixels[0]) > 10:
                # 计算倾斜角度（通过线性回归）
                try:
                    # 使用最小二乘法拟合
                    coeffs = np.polyfit(white_pixels[0], white_pixels[1], 1)
                    slope = coeffs[0]
                    # 如果斜率明显（倾斜角度大于5度），进行矫正
                    angle = np.arctan(slope) * 180 / np.pi
                    if abs(angle) > 5 and abs(angle) < 45:
                        # 创建仿射变换矩阵进行倾斜矫正
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, -angle * 0.5, 1.0)  # 部分矫正
                        digit_cropped = cv2.warpAffine(digit_cropped, M, (w, h), 
                                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                except:
                    pass

        # 重新找边界（矫正后可能变化）
        rows = np.any(digit_cropped > 0, axis=1)
        cols = np.any(digit_cropped > 0, axis=0)
        if np.any(rows) and np.any(cols):
            y_min2, y_max2 = np.where(rows)[0][[0, -1]]
            x_min2, x_max2 = np.where(cols)[0][[0, -1]]
            digit_cropped = digit_cropped[y_min2:y_max2 + 1, x_min2:x_max2 + 1]
            h, w = digit_cropped.shape

        # 调整大小，保持宽高比
        target_size = 20
        scale = min(target_size / h, target_size / w)
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))

        # 确保最小宽度
        if new_w < 6:
            new_w = 6

        # 调整大小
        if new_h > 0 and new_w > 0:
            digit_resized = cv2.resize(digit_cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            digit_resized = digit_cropped

        # 创建28x28的黑色背景
        digit_28x28 = np.zeros((28, 28), dtype=np.uint8)

        # 计算居中位置
        y_offset = (28 - new_h) // 2
        x_offset = (28 - new_w) // 2

        # 将数字放在中心
        if y_offset >= 0 and x_offset >= 0 and y_offset + new_h <= 28 and x_offset + new_w <= 28:
            digit_28x28[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = digit_resized

        return Image.fromarray(digit_28x28)

    def segment_digits(self, img, debug=False):
        """分割学号中的各个数字 - 改进版，专门处理细长的数字1"""
        # 转换为灰度图
        if isinstance(img, np.ndarray):
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
        else:
            gray = np.array(img.convert('L'))

        height, width = gray.shape

        if debug:
            print(f"原始图片尺寸: {width} x {height}")

        # 图片预处理：增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 轻微的高斯模糊去噪
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # 二值化 - 使用自适应阈值，对细线条更友好
        binary = cv2.adaptiveThreshold(
            gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10
        )

        # 形态学操作：轻微膨胀，连接断开的笔画（特别是细的数字1）
        kernel_dilate = np.ones((2, 2), np.uint8)
        binary = cv2.dilate(binary, kernel_dilate, iterations=1)

        # 去除小噪点
        kernel_open = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

        if debug:
            foreground_ratio = np.sum(binary > 0) / binary.size
            print(f"二值化完成，前景像素比例: {foreground_ratio:.2%}")
            # 保存二值化结果
            cv2.imwrite('debug_binary.png', binary)
            print("二值化图像已保存: debug_binary.png")

        # 使用轮廓检测方法分割数字
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            if debug:
                print("警告: 未找到任何轮廓")
            return [], []

        # 过滤和排序轮廓 - 降低面积阈值以检测细长的数字1
        digit_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            rect_area = w * h  # 边界框面积

            # 更宽松的过滤条件
            # 最小面积：降低阈值以检测细线条
            min_area = max(50, (width * height) / 20000)
            max_area = (width * height) / 2

            # 宽高比：允许非常细长的数字（如1）
            aspect_ratio = h / w if w > 0 else 0

            # 最小尺寸要求
            min_height = height * 0.1  # 至少是图片高度的10%
            min_width = 2  # 最小宽度2像素

            if debug and area > min_area / 2:
                print(f"  轮廓: 位置=({x}, {y}), 尺寸=({w}, {h}), 面积={area:.0f}, 宽高比={aspect_ratio:.2f}")

            if (area > min_area and area < max_area and
                    w >= min_width and h >= min_height and
                    aspect_ratio > 0.3 and aspect_ratio < 15):  # 更宽松的宽高比范围
                digit_contours.append((x, y, w, h, area, contour))

        # 按x坐标排序（从左到右）
        digit_contours.sort(key=lambda c: c[0])

        if debug:
            print(f"找到 {len(digit_contours)} 个有效数字轮廓")

        # 检查是否有遗漏的数字（通过分析间距）
        if len(digit_contours) >= 2:
            # 计算相邻数字的平均间距
            gaps = []
            for i in range(len(digit_contours) - 1):
                gap = digit_contours[i + 1][0] - (digit_contours[i][0] + digit_contours[i][2])
                gaps.append(gap)

            if gaps:
                avg_gap = np.mean(gaps)
                # 检查是否有异常大的间距（可能遗漏了数字）
                for i, gap in enumerate(gaps):
                    if gap > avg_gap * 2.5 and gap > 20:  # 间距异常大
                        if debug:
                            print(f"  警告: 数字 {i + 1} 和 {i + 2} 之间间距异常大 ({gap:.0f} vs 平均 {avg_gap:.0f})")

        # 提取每个数字
        digits = []
        digit_boxes = []

        for i, (x, y, w, h, area, contour) in enumerate(digit_contours):
            # 添加边距
            padding_x = max(3, int(w * 0.1))
            padding_y = max(3, int(h * 0.05))
            x_start = max(0, x - padding_x)
            y_start = max(0, y - padding_y)
            x_end = min(width, x + w + padding_x)
            y_end = min(height, y + h + padding_y)

            # 提取数字区域（使用原始灰度图）
            digit_region = gray[y_start:y_end, x_start:x_end]

            # 确保区域有效
            if digit_region.size == 0:
                continue

            # 转换为PIL Image
            digit_img = Image.fromarray(digit_region)
            digits.append(digit_img)
            digit_boxes.append((x, y, w, h))

            if debug:
                print(f"  数字 {i + 1}: 位置=({x}, {y}), 尺寸=({w}, {h}), 面积={area:.0f}, 宽高比={h/w:.2f}")

        if debug:
            print(f"最终分割出 {len(digits)} 个数字")
            # 保存分割后的数字图片用于调试
            try:
                debug_dir = 'debug_digits'
                os.makedirs(debug_dir, exist_ok=True)
                for i, digit_img in enumerate(digits):
                    digit_img.save(f'{debug_dir}/digit_{i + 1}.png')
                print(f"分割后的数字图片已保存到 {debug_dir}/ 目录")
            except Exception as e:
                if debug:
                    print(f"保存调试图片时出错: {e}")

        return digits, digit_boxes


def train_on_mnist():
    """使用MNIST数据集训练模型"""

    # 数据增强 - 添加更多变换以提高泛化能力
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),  # 增加旋转角度范围 (-15 到 15 度)
        transforms.RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),  # 增加平移范围
            scale=(0.85, 1.15),  # 增加缩放范围（有些人写的字很大，有些很小）
            shear=10  # 增加剪切变换（模拟斜体字）！！！这对识别7很重要
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    # 加载完整的测试集（10000个样本）
    test_dataset_full = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )

    # 将10000个测试样本划分为：5000个验证集 + 5000个测试集
    val_size = 5000
    test_size = 5000

    # 使用random_split进行随机划分
    val_dataset, test_dataset = random_split(
        test_dataset_full,
        [val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 设置随机种子以确保可重复性
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0  # Windows上建议设为0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    # 训练模型
    recognizer = StudentIDRecognition()
    recognizer.train(train_loader, val_loader=val_loader, epochs=15)  # 增加训练轮数

    # 训练完成后，在测试集上进行最终评估
    print("\n" + "=" * 50)
    print("在测试集上进行最终评估...")
    print("=" * 50)
    test_accuracy = recognizer.validate(test_loader)
    print(f"测试集准确率: {test_accuracy:.2f}%")
    print("=" * 50)

    # 如果训练过程中没有保存，最后保存一次
    if not os.path.exists('digit_model.pth'):
        torch.save(recognizer.model.state_dict(), 'digit_model.pth')
        print("模型已保存")

    return recognizer


def main():
    """主函数"""
    import cv2

    model_path = 'digit_model.pth'

    # 如果需要强制重新训练，可以删除模型文件或设置 force_retrain=True
    force_retrain = False  # 使用已训练好的模型

    # 如果模型已存在且不强制重新训练，直接加载；否则训练新模型
    if os.path.exists(model_path) and not force_retrain:
        print("加载已保存的模型...")
        recognizer = StudentIDRecognition(model_path=model_path)
    else:
        if force_retrain:
            print("强制重新训练模型...")
        else:
            print("模型不存在，开始训练新模型...")
        recognizer = train_on_mnist()

    # 识别学号
    image_path = 'Screenshot_20251227_012100.jpg'
    if os.path.exists(image_path):
        print("=" * 50)
        print("学号识别任务")
        print("=" * 50)
        print(f"任务输入: 学号照片 ({image_path})")
        print(f"训练集: MNIST")
        print(f"开始识别...")
        print("=" * 50)

        student_id = recognizer.recognize_student_id(image_path, debug=True)

        print("=" * 50)
        print("任务输出: 学号")
        print(f"识别的学号: {student_id}")
        print("=" * 50)
    else:
        print(f"图片文件不存在: {image_path}")
        print("请将学号照片放在当前目录下，文件名: qq_pic_merged_1766765458103.jpg")


if __name__ == "__main__":
    main()
