import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

def manual_convolution(image, kernel):
    """手动实现卷积操作"""
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # 填充边界 (Zero padding)
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(image, dtype=np.float32)

    # 卷积循环
    for i in range(h):
        for j in range(w):
            region = padded_img[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel)

    return np.clip(output, 0, 255).astype(np.uint8)


def manual_histogram(image):
    """手动计算直方图（灰度图）"""
    hist = np.zeros(256, dtype=int)
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            hist[image[i, j]] += 1
    return hist


def manual_color_histogram(image):
    """手动计算彩色图像的颜色直方图（RGB三个通道）"""
    h, w, c = image.shape
    hist_r = np.zeros(256, dtype=int)
    hist_g = np.zeros(256, dtype=int)
    hist_b = np.zeros(256, dtype=int)
    
    for i in range(h):
        for j in range(w):
            hist_b[image[i, j, 0]] += 1  # B通道
            hist_g[image[i, j, 1]] += 1  # G通道
            hist_r[image[i, j, 2]] += 1  # R通道
    
    return hist_r, hist_g, hist_b


def manual_mean(image):
    """手动计算均值"""
    h, w = image.shape
    total = 0
    count = 0
    for i in range(h):
        for j in range(w):
            total += image[i, j]
            count += 1
    return total / count if count > 0 else 0


def manual_variance(image, mean):
    """手动计算方差"""
    h, w = image.shape
    total = 0
    count = 0
    for i in range(h):
        for j in range(w):
            diff = image[i, j] - mean
            total += diff * diff
            count += 1
    return total / count if count > 0 else 0


def extract_texture_glcm(image):
    """
    手动实现纹理特征提取
    基于灰度共生矩阵原理的简化版，计算统计特征
    """
    # 手动计算均值
    mean = manual_mean(image)
    
    # 手动计算方差
    variance = manual_variance(image, mean)
    
    # 计算能量（归一化的灰度值平方和）
    h, w = image.shape
    energy = 0
    count = 0
    for i in range(h):
        for j in range(w):
            normalized = image[i, j] / 255.0
            energy += normalized * normalized
            count += 1
    energy = energy / count if count > 0 else 0
    
    # 计算对比度（基于相邻像素的差异）
    contrast = 0
    for i in range(h - 1):
        for j in range(w - 1):
            diff = abs(int(image[i, j]) - int(image[i, j + 1]))
            contrast += diff * diff
            diff = abs(int(image[i, j]) - int(image[i + 1, j]))
            contrast += diff * diff
    contrast = contrast / ((h - 1) * (w - 1) * 2) if (h > 1 and w > 1) else 0
    
    return np.array([mean, variance, energy, contrast])


# --- 主流程 ---
# 1. 读取图像（彩色和灰度）
image_path = '2f6be910fecf44bba3526ac90a285064.png'  # 请替换为你拍摄的图片路径
img_color = cv2.imread(image_path)
if img_color is None:
    raise FileNotFoundError(f"无法读取图像: {image_path}")

# 转换为灰度图用于滤波
img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# 2. 定义卷积核
# Sobel X方向
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
# Sobel Y方向
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
# 题目给定的卷积核
custom_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)

# 3. 执行滤波（手动实现，不调用函数包）
print("执行Sobel滤波...")
img_sobel_x = manual_convolution(img, sobel_x)
img_sobel_y = manual_convolution(img, sobel_y)
# Sobel 最终结果应该是 sqrt(x^2 + y^2)
img_sobel = np.sqrt(img_sobel_x.astype(float)**2 + img_sobel_y.astype(float)**2)
img_sobel = np.clip(img_sobel, 0, 255).astype(np.uint8)

print("执行自定义卷积核滤波...")
img_custom = manual_convolution(img, custom_kernel)

# 4. 计算直方图（手动实现）
print("计算灰度直方图...")
hist_gray = manual_histogram(img)

print("计算颜色直方图...")
hist_r, hist_g, hist_b = manual_color_histogram(img_color)

# 5. 提取纹理特征并保存（手动实现，不调用函数包）
print("提取纹理特征...")
texture_features = extract_texture_glcm(img)
np.save('texture_features.npy', texture_features)
print(f"纹理特征已保存: {texture_features}")
print(f"  均值: {texture_features[0]:.2f}")
print(f"  方差: {texture_features[1]:.2f}")
print(f"  能量: {texture_features[2]:.4f}")
print(f"  对比度: {texture_features[3]:.2f}")

# 6. 保存滤波后的图像
cv2.imwrite('sobel_filtered.jpg', img_sobel)
cv2.imwrite('custom_kernel_filtered.jpg', img_custom)
print("滤波结果已保存: sobel_filtered.jpg, custom_kernel_filtered.jpg")

# 7. 可视化结果
plt.figure(figsize=(15, 10))

# 显示原始图像和滤波结果
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title('原始图像 (Original)', fontsize=12)
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(img_sobel, cmap='gray')
plt.title('Sobel算子滤波结果', fontsize=12)
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(img_custom, cmap='gray')
plt.title('给定卷积核滤波结果', fontsize=12)
plt.axis('off')

# 显示灰度直方图
plt.subplot(2, 3, 4)
plt.bar(range(256), hist_gray, color='gray', alpha=0.7)
plt.title('灰度直方图 (Gray Histogram)', fontsize=12)
plt.xlabel('灰度值')
plt.ylabel('频数')

# 显示颜色直方图
plt.subplot(2, 3, 5)
plt.plot(range(256), hist_r, 'r', label='Red', alpha=0.7)
plt.plot(range(256), hist_g, 'g', label='Green', alpha=0.7)
plt.plot(range(256), hist_b, 'b', label='Blue', alpha=0.7)
plt.title('颜色直方图 (Color Histogram)', fontsize=12)
plt.xlabel('像素值')
plt.ylabel('频数')
plt.legend()

# 显示纹理特征
plt.subplot(2, 3, 6)
feature_names = ['均值', '方差', '能量', '对比度']
plt.bar(feature_names, texture_features, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
plt.title('纹理特征 (Texture Features)', fontsize=12)
plt.ylabel('特征值')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('experiment1_results.png', dpi=300, bbox_inches='tight')
print("可视化结果已保存: experiment1_results.png")
plt.show()