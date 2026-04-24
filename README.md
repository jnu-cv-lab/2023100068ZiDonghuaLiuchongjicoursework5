# 2023100068ZiDonghuaLiuchongjicoursework5
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- 1. 第一步：自己构造测试图 ----------------------
def create_test_image():
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    # 1. 画矩形
    cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 0), 2)
    # 2. 画圆
    cv2.circle(img, (550, 200), 80, (0, 255, 0), 2)
    # 3. 画多条平行线（水平）
    for y in [380, 430, 480]:
        cv2.line(img, (50, y), (750, y), (0, 0, 255), 2)
    # 4. 画垂直线，与水平线垂直
    for x in [200, 400, 600]:
        cv2.line(img, (x, 350), (x, 510), (0, 0, 255), 2)
    return img

# ---------------------- 2. 相似变换（缩放+旋转+平移，保形状、保角度） ----------------------
def similar_transform(img, angle=30, scale=0.8):
    h, w = img.shape[:2]
    center = (w//2, h//2)
    # 构造相似变换矩阵
    M = cv2.getRotationMatrix2D(center, angle, scale)
    res = cv2.warpAffine(img, M, (w, h), borderValue=(255,255,255))
    return res

# ---------------------- 3. 仿射变换（任意拉伸、倾斜，保平行） ----------------------
def affine_transform(img):
    h, w = img.shape[:2]
    # 原图3个点
    pts1 = np.float32([[50,50], [200,50], [50,200]])
    # 变换后对应3个点
    pts2 = np.float32([[70,80], [220,60], [90,230]])
    M = cv2.getAffineTransform(pts1, pts2)
    res = cv2.warpAffine(img, M, (w, h), borderValue=(255,255,255))
    return res

# ---------------------- 4. 透视变换（投影变换，仅保共线性） ----------------------
def perspective_transform(img):
    h, w = img.shape[:2]
    # 原图4个角点
    pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    # 透视畸变后的4个点，模拟斜着拍照
    pts2 = np.float32([[60,30], [w-90,60], [20,h-40], [w-30,h-20]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    res = cv2.warpPerspective(img, M, (w, h), borderValue=(255,255,255))
    return res

# ---------------------- 5. 透视畸变校正 ----------------------
def perspective_correction(distort_img, src_pts, dst_w, dst_h):
    # 手动选取畸变图4个角，输出规整矩形
    pts_std = np.float32([[0,0], [dst_w,0], [0,dst_h], [dst_w,dst_h]])
    M = cv2.getPerspectiveTransform(src_pts, pts_std)
    corrected = cv2.warpPerspective(distort_img, M, (dst_w, dst_h))
    return corrected

# ---------------------- 6. 主程序 & 几何性质总结 ----------------------
if __name__ == "__main__":
    # 生成测试原图
    original = create_test_image()

    # 执行三类变换
    img_similar = similar_transform(original)
    img_affine = affine_transform(original)
    img_perspective = perspective_transform(original)

    # 绘图对比
    plt.figure(figsize=(16,10))
    plt.subplot(221),plt.imshow(cv2.cvtColor(original,cv2.COLOR_BGR2RGB)),plt.title("原图")
    plt.subplot(222),plt.imshow(cv2.cvtColor(img_similar,cv2.COLOR_BGR2RGB)),plt.title("相似变换")
    plt.subplot(223),plt.imshow(cv2.cvtColor(img_affine,cv2.COLOR_BGR2RGB)),plt.title("仿射变换")
    plt.subplot(224),plt.imshow(cv2.cvtColor(img_perspective,cv2.COLOR_BGR2RGB)),plt.title("透视变换")
    plt.tight_layout()
    plt.show()

    # ========== 几何性质总结表格 ==========
    print("======= 几何变换性质对比总结 =======")
    print("1. 直线是否保持为直线：")
    print("   相似变换：✅ 始终保持直线")
    print("   仿射变换：✅ 始终保持直线")
    print("   透视变换：✅ 始终保持直线")
    print("\n2. 平行线是否保持平行：")
    print("   相似变换：✅ 平行线永远平行")
    print("   仿射变换：✅ 平行线永远平行")
    print("   透视变换：❌ 平行线会汇聚、不再平行")
    print("\n3. 垂直直线是否保持垂直：")
    print("   相似变换：✅ 角度不变，保持垂直")
    print("   仿射变换：❌ 角度改变，不一定垂直")
    print("   透视变换：❌ 几乎不再保持垂直")
    print("\n4. 圆是否保持为圆：")
    print("   相似变换：✅ 圆始终是圆")
    print("   仿射变换：❌ 圆会变成椭圆")
    print("   透视变换：❌ 圆会变为任意椭圆/变形图形")

    # ---------------------- 7. 实拍A4纸透视矫正演示 ----------------------
    # 1. 请提前自己拍一张斜着拍的、带文字表格的A4纸，命名为 "paper.jpg" 放在同目录
    # 2. 手动用鼠标标注A4纸的四个角，替换下面坐标
    print("\n======= 透视畸变校正演示 =======")
    try:
        paper = cv2.imread("paper.jpg")
        ph, pw = paper.shape[:2]
        # 示例：手动选取纸张四个角（左上、右上、左下、右下），请根据自己图片修改
        paper_pts = np.float32([[120,90], [pw-100,110], [80,ph-80], [pw-70,ph-50]])
        # 矫正输出标准A4比例图像
        corrected_paper = perspective_correction(paper, paper_pts, 500, 700)
        plt.figure(figsize=(10,5))
        plt.subplot(121),plt.imshow(cv2.cvtColor(paper,cv2.COLOR_BGR2RGB)),plt.title("畸变原图")
        plt.subplot(122),plt.imshow(cv2.cvtColor(corrected_paper,cv2.COLOR_BGR2RGB)),plt.title("矫正后")
        plt.show()
        print("矫正完成✅，矫正后文字、表格方正，几乎无变形！")
    except:
        print("未找到paper.jpg，你可以放入自己拍摄的A4照片后运行矫正功能")
