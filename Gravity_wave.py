# -*- coding: cp932 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 重力波による画像の歪みをシミュレーションする関数
def apply_gravitational_wave_distortion(img, time, amplitude=0.5, omega=0.04, C=0):
    """画像に重力波による歪みを適用（画像の中心を振動の中心とする）
    Args:
        img: 入力画像 (numpy array)
        time: アニメーションにおける現在の時間ステップ
        amplitude: 歪みの振幅
        omega: 角振動数
        C: 位相定数
    """
    rows, cols = img.shape[:2]
    center_x, center_y = cols // 2, rows // 2  # 画像中心の座標
    distorted_img = np.zeros_like(img)

    # 各画素に対してx, y方向の歪みを適用
    for y in range(rows):
        for x in range(cols):
            # 中心からの相対位置 (dx, dy) を計算
            dx = x - center_x
            dy = y - center_y
            distance_from_center = np.sqrt(dx**2 + dy**2)

            # x方向とy方向の歪みを計算
            displacement_x = int( - amplitude/2 * np.cos(2*np.pi*omega*time) * dx)
            displacement_y = int( + amplitude/2 * np.cos(2*np.pi*omega*time) * dy)
            
            # 新しい位置に色を移動させる
            new_x = min(max(x + displacement_x, 0), cols - 1)
            new_y = min(max(y + displacement_y, 0), rows - 1)
            distorted_img[y, x] = img[new_y, new_x]

    return distorted_img

# 画像を読み込む
img_path = 'runa.jpg'  # 任意の画像ファイルパス
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# アニメーションを作成するための準備
fig, ax = plt.subplots()
ax.axis('off')
im = ax.imshow(img)

# アニメーションの更新関数
def update(frame):
    distorted_img = apply_gravitational_wave_distortion(img, time=frame)
    im.set_array(distorted_img)
    return [im]

# アニメーションの作成
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)

# アニメーションの表示
plt.show()
