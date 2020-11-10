# LaneDetectionInFog
## Dehaze: 去雾      
   adjust.m : 在对比度上优化图像     
   boxfilter.m:  在导向滤波中引用       
   dehaze.m: 去雾副函数      
   dehazing.m： 去雾主函数        
   estimate_airlight.m： 估算大气光，偏振对雾况敏感，若其估算的大气光图不准确，可辅助计算        
   guidedfilter.m : 导向滤波        
   PolarizationDefog.m：程序入口     
   wls_optimization.m： 白平衡优化传输图     
## LaneDetection: 车道线识别
   preprocess.m： 直道车道线检测预处理     
   LaneDetect.m： 霍夫变换检测车道线  
   track_straight.m： 直道车道线跟踪        
   SlidingWindow.py： 滑动窗口检测车道线       
   track_window.py： 基于滑动窗口的车道跟踪
   
