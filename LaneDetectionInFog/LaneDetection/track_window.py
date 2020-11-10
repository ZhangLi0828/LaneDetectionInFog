import numpy as np
import utils


# 跟踪算法
class Line():
    def __init__(self):
        #self.recent_fit = [0, 0]
        #self.best_fit = [0, 0]
        self.left_fit = np.array([0,0])
        self.right_fit = np.array([0,0])
        self.left_base = 0
        self.right_base = 0
        self.mode = None
        self.find = True
        self.track = False
        self.left_diff = None
        self.right_diff = None
    
    def edge_check(self,edge):    # 返回四个值
        histogram = np.sum(edge, axis=0)

        # import matplotlib.pyplot as plt
        # plt.plot(histogram)
        # plt.show()

        # 中点位置
        midpoint = np.int(histogram.shape[0] / 2)

        left_sum = int(max(histogram[:midpoint]))
        right_sum = int(max(histogram[midpoint:]))

        left_base = np.argmax(histogram[:midpoint])  # 左基点
        right_base = np.argmax(histogram[midpoint:]) + midpoint

        print("left_sum= ", int(left_sum))
        print("right_sum = ", int(right_sum))

        dis = int(right_base - left_base)  # 像素距离
        print("lane_distance= ", dis)

        
        return left_sum, right_sum, left_base, right_base



    def mode_check(self, left_sum, right_sum, left_base, right_base):   # draw模式
        '''
        if distance is not None:
            if distance < 120:
                self.mode = 0      # detection failed

            else:  # distance >120
                if (left_sum < 500 and right_sum < 500):
                    self.mode = 0

                elif left_sum >= 500 and right_sum < 500 :
                    self.mode = 1    # draw single line, no area

                elif right_sum >= 500 and left_sum < 500 :
                    self.mode = 2    # draw single line, no area

                else:   # right_sum >= 500 and left_sum >= 500
                    if distance > 200:
                        self.mode = 3    # double lane, draw both lines , no rea,

                    elif 120 <= distance <= 150:
                        self.mode = 4    # draw area
                        self.current_fit = [left_fit, right_fit]

                    elif 150 < distance <= 170:   # 按上一帧修正
                        self.mode = 4   # draw area
                        left_fit = self.current_fit[0]
                        right_fit = self.current_fit[1]

                    else:
                        self.mode = 0


        return self.mode, left_fit, right_fit
        '''
        
        distance = right_base - left_base
        self.left_diff = abs(left_base - self.left_base)
        self.right_diff = abs(right_base - self.right_base)

        if self.left_base == 0 and self.right_base == 0:
            self.mode = 3
            self.find = True

        if left_sum < 50 and right_sum < 50:
            self.mode = 0

        elif self.left_diff < 50 or self.right_diff < 50 :
            self.mode = 3
            self.find = False    # 跟踪

        else:
            self.find = True
            if distance < 180 or distance > 250 :
                self.mode = 0      # 跟踪失败

            if left_sum >= 50 and right_sum < 50 :
                self.mode = 1    # 检测出左车道线

            if right_sum >= 50 and left_sum < 50 :
                self.mode = 2    # 检测出右车道线


    def update(self, left_fit, right_fit, left_base, right_base):   # 更新
        if left_fit is not None:
            if self.mode == 3 or self.track:
                if self.left_diff < 50:
                    self.left_base = left_base
                if self.right_diff < 50:
                    self.right_base = right_base

                self.left_fit = left_fit
                self.right_fit = right_fit
        print("left_base=", self.left_base)
        print("right_base=", self.right_base)
        print("left_fit=", self.left_fit)
        print("right_fit=", self.right_fit)




if __name__ == "__main__":
    import time
    import os
    import cv2
    import utils

    time1 = time.time()
    from skimage import morphology

    path = "perspective"
    sum = len(os.listdir(path))
    write_path = "edge"
    for c in range(1, sum + 1):
        item = path + '//frame' + str(c) + '.jpg'
        bird = cv2.imread(item)
        mask = np.ones_like(bird)
        filter = np.max(bird) * 0.2
        mask[bird < filter] = 0

        edge = utils.thresholding(bird)
        bird = cv2.medianBlur(bird, 9)
        # 去连通域
        edge = morphology.remove_small_objects(edge.astype('bool'), min_size=80, connectivity=2, in_place=True)
        edge = np.multiply(edge, mask)

        cv2.imwrite(write_path + '//frame' + str(c) + '.jpg', edge*255)

    time2 = time.time()
    print("用时：", np.round((time2 - time1) / 60, 2), "min")