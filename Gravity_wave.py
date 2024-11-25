# -*- coding: cp932 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# �d�͔g�ɂ��摜�̘c�݂��V�~�����[�V��������֐�
def apply_gravitational_wave_distortion(img, time, amplitude=0.5, omega=0.04, C=0):
    """�摜�ɏd�͔g�ɂ��c�݂�K�p�i�摜�̒��S��U���̒��S�Ƃ���j
    Args:
        img: ���͉摜 (numpy array)
        time: �A�j���[�V�����ɂ����錻�݂̎��ԃX�e�b�v
        amplitude: �c�݂̐U��
        omega: �p�U����
        C: �ʑ��萔
    """
    rows, cols = img.shape[:2]
    center_x, center_y = cols // 2, rows // 2  # �摜���S�̍��W
    distorted_img = np.zeros_like(img)

    # �e��f�ɑ΂���x, y�����̘c�݂�K�p
    for y in range(rows):
        for x in range(cols):
            # ���S����̑��Έʒu (dx, dy) ���v�Z
            dx = x - center_x
            dy = y - center_y
            distance_from_center = np.sqrt(dx**2 + dy**2)

            # x������y�����̘c�݂��v�Z
            displacement_x = int( - amplitude/2 * np.cos(2*np.pi*omega*time) * dx)
            displacement_y = int( + amplitude/2 * np.cos(2*np.pi*omega*time) * dy)
            
            # �V�����ʒu�ɐF���ړ�������
            new_x = min(max(x + displacement_x, 0), cols - 1)
            new_y = min(max(y + displacement_y, 0), rows - 1)
            distorted_img[y, x] = img[new_y, new_x]

    return distorted_img

# �摜��ǂݍ���
img_path = 'runa.jpg'  # �C�ӂ̉摜�t�@�C���p�X
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# �A�j���[�V�������쐬���邽�߂̏���
fig, ax = plt.subplots()
ax.axis('off')
im = ax.imshow(img)

# �A�j���[�V�����̍X�V�֐�
def update(frame):
    distorted_img = apply_gravitational_wave_distortion(img, time=frame)
    im.set_array(distorted_img)
    return [im]

# �A�j���[�V�����̍쐬
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)

# �A�j���[�V�����̕\��
plt.show()
