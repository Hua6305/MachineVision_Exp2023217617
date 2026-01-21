import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置绘图字体
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


class LaneDetection:
    def __init__(self):
        """初始化车道线检测器"""
        self.left_line_history = []
        self.right_line_history = []
        self.history_length = 10

    def color_filter(self, img):
        """
        颜色过滤：大幅放宽阈值，适应灰暗路面或不明显的线
        """
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        # 【修改1】白色阈值下限降至 120 (原160/180)
        # 这样能捕捉到比较暗的白色车道线
        white_lower = np.array([0, 120, 0])
        white_upper = np.array([180, 255, 255])
        white_mask = cv2.inRange(hls, white_lower, white_upper)

        # 【修改2】黄色阈值放宽
        yellow_lower = np.array([10, 30, 100])
        yellow_upper = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)

        color_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # 形态学操作去噪
        kernel = np.ones((3, 3), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

        return color_mask

    def region_of_interest(self, img):
        """定义感兴趣区域"""
        height, width = img.shape[:2]

        # 定义梯形区域，覆盖大部分车道
        vertices = np.array([[
            (int(width * 0.05), height),  # 左下 (放宽到边缘)
            (int(width * 0.45), int(height * 0.4)),  # 左上
            (int(width * 0.55), int(height * 0.4)),  # 右上
            (int(width * 0.95), height)  # 右下 (放宽到边缘)
        ]], dtype=np.int32)

        mask = np.zeros_like(img)
        cv2.fillPoly(mask, [vertices], 255)
        masked_img = cv2.bitwise_and(img, mask)

        return masked_img

    def edge_detection(self, img, color_mask):
        """边缘检测"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        combined = cv2.bitwise_and(edges, color_mask)
        color_edges = cv2.Canny(color_mask, 50, 150)
        final_edges = cv2.bitwise_or(combined, color_edges)

        return final_edges

    def detect_lines(self, edges, img_shape):
        """
        霍夫变换：使用高灵敏度参数
        """
        height, width = img_shape[:2]

        # 【修改3】参数调整为高灵敏度
        rho = 1
        theta = np.pi / 180
        threshold = 20  # 只要20个点共线就认为是线 (原50)
        min_line_length = 20  # 允许更短的线段
        max_line_gap = 150  # 允许更大的断裂连接

        lines = cv2.HoughLinesP(
            edges,
            rho=rho,
            theta=theta,
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )
        return lines

    def classify_lines(self, lines, img_shape):
        """
        【关键修改】基于位置的严格分类，防止交叉
        """
        height, width = img_shape[:2]
        left_lines = []
        right_lines = []

        if lines is None:
            return left_lines, right_lines

        center_x = width // 2

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 排除几乎垂直或水平的噪点
            if x2 == x1: continue
            slope = (y2 - y1) / (x2 - x1)

            # 1. 斜率过滤 (排除水平线)
            if abs(slope) < 0.3: continue

            # 2. 【核心逻辑】位置约束
            # 左线：斜率为负，且必须都在图像左侧
            if slope < 0 and x1 < center_x and x2 < center_x:
                left_lines.append((line[0], slope, 1))

            # 右线：斜率为正，且必须都在图像右侧
            elif slope > 0 and x1 > center_x and x2 > center_x:
                right_lines.append((line[0], slope, 1))

        return left_lines, right_lines

    def fit_lane_line(self, lines, img_shape):
        """拟合车道线"""
        if len(lines) == 0:
            return None

        height, width = img_shape[:2]
        x_coords = []
        y_coords = []

        for line_data in lines:
            line, slope, _ = line_data
            x1, y1, x2, y2 = line
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])

        if len(x_coords) < 2:
            return None

        try:
            # 线性拟合
            coeffs = np.polyfit(y_coords, x_coords, 1)

            y1 = height
            y2 = int(height * 0.6)  # 延伸到图像中部

            x1 = int(np.polyval(coeffs, y1))
            x2 = int(np.polyval(coeffs, y2))

            return [x1, y1, x2, y2]
        except:
            return None

    def draw_lane(self, img, left_line, right_line):
        """
        绘制：使用绿色加粗线条覆盖车道
        """
        result = img.copy()

        # 1. 绘制中间的透明区域
        if left_line is not None and right_line is not None:
            overlay = img.copy()
            pts = np.array([
                [left_line[0], left_line[1]],
                [left_line[2], left_line[3]],
                [right_line[2], right_line[3]],
                [right_line[0], right_line[1]]
            ], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            result = cv2.addWeighted(overlay, 0.2, result, 0.8, 0)

        # 2. 【修改4】绘制车道线：绿色，宽度10
        if left_line is not None:
            cv2.line(result, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 0), 10)

        if right_line is not None:
            cv2.line(result, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 10)

        return result

    def process_image(self, image_path):
        """处理单张图片"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return None, None, None

        # 清除历史
        self.left_line_history = []
        self.right_line_history = []

        # 流程
        color_mask = self.color_filter(img)
        edges = self.edge_detection(img, color_mask)
        roi_edges = self.region_of_interest(edges)
        lines = self.detect_lines(roi_edges, img.shape)

        # 调试信息
        if lines is not None:
            print(f"检测到 {len(lines)} 条原始线段")
        else:
            print("未检测到线段，请检查阈值或图片路径")

        left_lines, right_lines = self.classify_lines(lines, img.shape)
        left_line = self.fit_lane_line(left_lines, img.shape)
        right_line = self.fit_lane_line(right_lines, img.shape)

        result = self.draw_lane(img, left_line, right_line)
        return result, left_line, right_line

    def process_video(self, video_path, output_path=None, display=True):
        """视频处理函数 (保持不变，省略以节省空间，功能与process_frame联动)"""
        # 如果你需要运行视频，请把原来 2.py 中的 process_video 和 process_frame 复制回来
        # 这里为了确保你能直接运行图片检测，简化了这部分
        pass


def main():
    detector = LaneDetection()

    # 请确认这里的文件名是否与你上传的一致
    image_path = '1AE957599EE32007878B9E5693D251D3.jpg'

    if os.path.exists(image_path):
        print(f"正在处理: {image_path}")
        result, left, right = detector.process_image(image_path)

        if result is not None:
            # 保存并显示结果
            save_name = 'lane_result_final.jpg'
            cv2.imwrite(save_name, result)
            print(f"处理成功，已保存为: {save_name}")

            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            plt.title('车道线检测 (Final Fixed)', fontsize=14)
            plt.axis('off')
            plt.show()

            # 打印坐标
            if left: print(f"左车道线: {left}")
            if right: print(f"右车道线: {right}")
        else:
            print("处理失败，未生成结果图像")
    else:
        print(f"错误：找不到文件 {image_path}")


if __name__ == "__main__":
    main()