import cv2
import numpy as np
import IPM
import line
import os


def getDoP(img):    #计算偏振度，可以不用
    H, W = img.shape
    img = img.astype(np.float32)/255  #float64
    I90 = img[:H//2, :W//2]
    I45 = img[:H//2, W//2:W]
    I0 = img[H//2:H, W//2:W]
    S0 = I0 + I90
    S1 = I0 - I90
    S2 = 2*I45 - S0
    np.seterr(divide='ignore', invalid='ignore')
    DoP = np.sqrt(S1*S1+S2*S2)/S0
    AoP = np.arctan(S2/S1)/2
    #I90_draw = frame[:H//2, :W//2]
    DoP = cv2.resize(DoP, (640, 480))
    AoP = cv2.resize(AoP, (640, 480))
    I90 = cv2.resize(I90, (640, 480))

    return  DoP,AoP,I90


def abs_sobel_thresh(img, orient='x', thresh_min=10, thresh_max=50):   #梯度阈值
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))

    scaled_sobel = np.uint8(255 * abs_sobel/np.max(abs_sobel))  # 梯度图
    #cv2.imshow("scaled_sobel", scaled_sobel)

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return binary_output

def mag_thresh(img, sobel_kernel=7, thresh=(10, 50)):    # 梯度幅值阈值

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    gradmag = np.sqrt(sobelx**2 + sobely**2)    #计算梯度幅值

    gradmag = np.uint8(255 * gradmag / np.max(gradmag))
    #cv2.imshow("gradmag",gradmag)   # 幅值图
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    return binary_output

def dir_threshold(img, sobel_kernel=7, thresh=(0, 0.15)):   # 梯度方向阈值

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1]) ] = 1

    return binary_output

'''
def gaussBlur(img, H=5, W=5, sigma_y=5, sigma_x=7,_boundry='fill', _fillvalue=0): #论文方法，无效果
    from scipy import signal
    #构建垂直方向高斯卷积核
    kernel_y = cv2.getGaussianKernel(H, sigma_y, cv2.CV_64F)

    #构建水平方向高斯卷积核
    kernel_x = np.zeros((W,1),np.float32)
    cW = (W-1)/2     #锚点
    sigma_xsqr = np.power(sigma_x,2)
    for w in range(W):
        norm = np.power(w-cW,2)
        kernel_x[w] = np.exp(-norm/(2*sigma_xsqr))*(sigma_xsqr-norm)/np.power(sigma_xsqr,2)
    kernel_x = kernel_x/np.sum(kernel_x)   #归一化，所以无1/(2*pi*sigma_sqr)
    kernel_x = np.transpose(kernel_x)

    gaussblur_y = signal.convolve2d(img, kernel_y, mode='same', boundary=_boundry, fillvalue=_fillvalue)
    gaussblur = signal.convolve2d(gaussblur_y, kernel_x, mode='same', boundary=_boundry, fillvalue=_fillvalue)
    gaussblur = np.round(gaussblur)
    gaussblur = gaussblur.astype(np.uint8)

    return gaussblur
'''

def thresholding(img):    # 三阈值过滤，获取边缘图
    x_grad = abs_sobel_thresh(img)
    grad_mag = mag_thresh(img)
    dir_thresh = dir_threshold(img)
    threshholded = np.zeros_like(x_grad)
    threshholded[((x_grad == 1) & (dir_thresh == 1))|((grad_mag == 1) & (dir_thresh == 1))] = 255
    #threshholded[(x_grad == 1) & (dir_thresh == 1) & (grad_mag == 1)] = 255
    return threshholded

def get_M():   #仿射变换
    src = np.float32([[(0, 320), (0, 460), (640, 320), (640, 460)]])
    dst = np.float32([[(0, 0),  (200, 480), (640, 0), (440, 480)]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv

def MahalanobisDist(x, y):   # 计算马氏距离
    covariance_xy = np.cov(x, y, rowvar=False)  #协方差矩阵
    inv_covariance_xy = np.linalg.inv(covariance_xy)  #协方差矩阵逆
    xy_mean = np.mean(x), np.mean(y)
    x_diff = np.array([x_i - xy_mean[0] for x_i in x])
    y_diff = np.array([y_i - xy_mean[1] for y_i in y])
    diff_xy = np.transpose([x_diff, y_diff])

    md = []
    for i in range(len(diff_xy)):
        md.append(np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]), inv_covariance_xy), diff_xy[i])))
    return md


def MD_removeOutliers(x, y, MD_thresh):   #按马氏距离剔除异常点，可以不用

    MD = MahalanobisDist(x, y)
    threshold = np.mean(MD) * MD_thresh
    nx, ny, outliers = [], [], []
    for i in range(len(MD)):
        if MD[i] <= threshold:
            nx.append(x[i])
            ny.append(y[i])
        else:
            outliers.append(i)
    return (nx, ny)

#def find_line(edge, left_base, right_base):
def find_line(edge):
    histogram = np.sum(edge, axis=0)

    #import matplotlib.pyplot as plt
    #plt.plot(histogram)
    #plt.show()

    # 中点位置
    midpoint = np.int(histogram.shape[0] / 2)

    #left_sum = int(max(histogram[:midpoint]))      #左边像素累计
    #right_sum = int(max(histogram[midpoint:]))     #右边像素累计


    left_base = np.argmax(histogram[:midpoint])  # 左基点
    right_base = np.argmax(histogram[midpoint:]) + midpoint
    dis = int(right_base - left_base)  # 像素距离

    #print("left_sum= ", int(left_sum))
    #print("right_sum = ", int(right_sum))
    #print("lane_distance= ", dis)

    nwindows = 10  # 窗口个数
    window_height = np.int(edge.shape[0] / nwindows)   # 窗高

    nonzero = edge.nonzero()   # 图中不为0元素的矩阵坐标
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    margin_out = 15        # 窗口宽度变量
    margin_in = 35
    minpix = 90              # 窗中最少像素值

    leftx_current = left_base + 5  # 开始滑动的位置
    rightx_current = right_base

    left_inds = []     # 坐标集合
    right_inds = []

    edge_for_windows = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR) # 为方框图，此处type(edge)=uint8

    for window in range(nwindows):  # 从下往上滑动

        win_y_high = edge.shape[0] - (window + 1) * window_height  #  窗口上界   从下往上滑
        win_y_low = edge.shape[0] - window * window_height       # 窗口下界

        #win_y_low = (window + 1) * window_height  # 窗口上界   从上往下滑
        #win_y_high = window * window_height       # 窗口下界

        win_xleft_low = leftx_current - margin_out    # 左上角坐标（左窗口）
        win_xleft_high = leftx_current + margin_in   # 右上角坐标
        win_xright_low = rightx_current - margin_in   # 左上角坐标（右窗口）
        win_xright_high = rightx_current + margin_out

        # 画滑动窗
        cv2.rectangle(edge_for_windows, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 0, 150), 2)
        cv2.rectangle(edge_for_windows, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (150, 0, 0), 2)

        # 在窗口内元素的坐标在nonzero中的位置索引
        good_left_inds = ((nonzeroy <= win_y_low) & (nonzeroy > win_y_high) &    # 注意>, <
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
                          ).nonzero()[0]
        good_right_inds = ((nonzeroy <= win_y_low) & (nonzeroy > win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
                           ).nonzero()[0]

        left_inds.append(good_left_inds)
        right_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:                           # recenter
            leftx_current = np.int(np.mean(nonzerox[good_left_inds])-2)
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    #cv2.imshow("windows.jpg", edge_for_windows)
    #cv2.waitKey()
    left_inds = np.concatenate(left_inds)
    right_inds = np.concatenate(right_inds)

    # 提取坐标位置
    leftx = nonzerox[left_inds]
    lefty = nonzeroy[left_inds]
    rightx = nonzerox[right_inds]
    righty = nonzeroy[right_inds]


    try:
        left_fit = np.array(np.polyfit(lefty, leftx, 2))   #直线拟合，返回k,b
        right_fit = np.array(np.polyfit(righty, rightx, 2))

        return left_fit, right_fit,edge_for_windows

    except:
        print("no lanes")
        return None, None,None


#==================================跟踪算法=====================================
def track(edge, left_fit, right_fit):   #每一帧都要检测边缘图，left_fit,right_fit为上一帧参数
    #本帧信息

    nonzero = edge.nonzero()   # edge图上非零像素坐标
    nonzeroy = np.array(nonzero[0])   # 非零像素对应纵坐标
    nonzerox = np.array(nonzero[1])   # 非零像素对应横坐标

    margin = 5

    left_lane_inds = ((nonzerox > (left_fit[0] * nonzeroy + left_fit[1] - margin)) &
                      (nonzerox < (left_fit[0] * nonzeroy + left_fit[1] + margin)))

    #在margin范围内搜索
    right_lane_inds = ((nonzerox > (right_fit[0] * nonzeroy + right_fit[1]) - margin) &
                       (nonzerox < (right_fit[0] * nonzeroy + right_fit[1] + margin)))

    # 提取待拟合点坐标
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    try:
        left_fit = np.polyfit(lefty, leftx, 1)
        right_fit = np.polyfit(righty, rightx, 1)
        print("track detected")
        return 1, left_fit, right_fit  #跟踪成功

    except:
        print("track undetected")
        return 0, left_fit, right_fit  #跟踪失败


def calculate_curv_and_pos(left_x, right_x):   #计算曲率半径，可以不用

    h = left_x.shape[0]
    lane_width = np.absolute(left_x[h-1] - right_x[h-1])
    lane_xm_per_pix = 3.5 / lane_width   # 单位：m/像素
    veh_pos = (left_x[h-1] + right_x[h-1]) / 2.
    cen_pos = 365 / 2.   # midpoint
    distance_from_center = (cen_pos - veh_pos) * lane_xm_per_pix
    return distance_from_center

def draw_values(img, distance_from_center):
    font = cv2.FONT_HERSHEY_SIMPLEX
    #radius_text = "R of lanes: %sm" % (round(curvature))

    if distance_from_center > 0:
        pos_flag = 'right'
    else:
        pos_flag = 'left'

    #cv2.putText(img, radius_text, (50, 30), font, 1, (150, 50, 100), 2)
    center_text = "%s to center: %.3fm " % (pos_flag, abs(distance_from_center))
    cv2.putText(img, center_text, (50, 50), font, 1, (0, 0, 150), 2)
    return img

def draw(frame, Minv, left_fit, right_fit, mode):

    H,W = np.shape(frame)
    ploty = np.linspace(0, H - 1, H)

    #left_x = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]  #直线上的点  shape=480
    #right_x = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    left_x = left_fit[0] * ploty + left_fit[1]
    right_x = right_fit[0] * ploty + right_fit[1]

    # 创建画线平面
    map = np.zeros((H, W, 3))
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR).astype(np.float64)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_x, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    if mode == 1:  # 画左线
        left_lane = cv2.polylines(map, np.int_(pts_left), False, (0, 0, 5), 15)
        warp = cv2.warpPerspective(left_lane, Minv, (W, H))
        result = cv2.addWeighted(frame, 1, warp, 1, 0)

    elif mode == 2:  # 画右线
        right_lane = cv2.polylines(map, np.int_(pts_right), False, (0, 0, 5), 15)
        warp = cv2.warpPerspective(right_lane, Minv, (W, H))
        result = cv2.addWeighted(frame, 1, warp, 1, 0)

    elif mode == 3:  # 画area
        lane_area = cv2.fillPoly(map, np.int_([pts]), (0, 4, 0))
        warp = cv2.warpPerspective(lane_area, Minv, (W, H))   # float64
        result = cv2.addWeighted(frame, 1, warp, 0.1, 0)

    else:
        result = frame

    return result, left_x, right_x


def video_cut(w, h, url_1, url_2):  # 读取视频
    # w,h:截取宽度，长度   i:截取倍数， url_1:视频地址
    cap = cv2.VideoCapture(url_1)  # 从文件读取视频
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取原视频的帧率
    size = (int(w), int(h))  # 自定义需要截取的画面的大小
    out = cv2.VideoWriter(url_2, fourcc, fps, size)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret==True:
            frame_90 = frame[0:h,0:w]  # 截取90度图
            out.write(frame_90)
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    #IPM = IPM.IPM()  # IPM变换

    #uvGrid = IPM.Ground2Image()  #俯视图范围
    #M,Minv = get_M()   #仿射变换矩阵
    frame = cv2.imread("bird.png")

    img = frame[:, :, 0]
    birdview = cv2.resize(img, (640, 480))
    #birdview = IPM.GetIPM(img, uvGrid)   # 640 * 480，获得IPM变换的俯视图

    edge = thresholding(birdview)   # 边缘图

    #=============================采用IPM变换，边缘图需如下处理===============================
    # ret, binary = cv2.threshold(birdview, 1, 255, cv2.THRESH_BINARY)
    # contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contour = np.ones((birdview.shape[0], birdview.shape[1], 3), dtype=np.uint8)  # 生成轮廓图
    # cv2.drawContours(contour, contours, -1, (255, 255, 255), 10)
    # edge = np.multiply(edge, contour[:, :, 0])
    #======================================================================================

    #left_fit, right_fit, distance, left_num, right_num = find_line(edge)

    #不采取跟踪，单帧检测
    #lane_area, left_x, right_x = draw(birdview, M,  left_fit, right_fit, 3)

    cv2.imshow("lane_Area", edge)
    cv2.waitKey()















