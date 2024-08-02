"""
step1:pip install easyocr
step2:install cuda and cudnn, put "export /usr/local/cuda-11.7/include/cudnn.h" in vim ~/.bashrc

"""

import cv2
import easyocr


# 定义一个红色字体的ANSI转义代码
RED_FONT = "\033[31m"
# 重置所有文本属性的ANSI转义代码
END = "\033[0m"

class SubtitleDetector:
    """
    det_time = 1 #自定义的检测时间
    set_dettime = False #True时按照opencv的帧数, 为false时为自定义的检测时间
    max_devloc = 50 #不同类型文本框沿y轴间的最大间距
    print_res = False  # 输出所有的检测结果
    print_time = False #输出时间, 单位秒(s)
    print_selectres = False  #输出选中box的检测结果
    print_totaloutlist = False  #输出所有的检测框和时间
    print_delrepeatoutlist = True  #输出去重复之后的检测框和时间
    """

    def __init__(
        self,
        vd_path,
        begin_t,
        end_t,
        det_time=1,
        set_dettime=False,
        max_devloc=50,
        print_res=False,
        print_time=False,
        print_selectres=False,
        print_totaloutlist=False,
        use_deltime=False,
        print_delrepeatoutlist=False,
    ):
        self.vd_path = vd_path
        self.begin_t = begin_t
        self.end_t = end_t
        self.det_time = det_time
        self.set_dettime = set_dettime
        self.max_devloc = max_devloc
        self.print_res = print_res
        self.print_time = print_time
        self.print_selectres = print_selectres
        self.print_totaloutlist = print_totaloutlist
        self.use_deltime = use_deltime
        self.print_delrepeatoutlist = print_delrepeatoutlist
        self.height = 0
        self.width = 0
        self.select_down = True

    def del_repeat(self, det_list):
        for i in range(len(det_list)):
            det_list[i].append(det_list[i][2] - det_list[i][1])
            for j in range(i+1, len(det_list) ):
                if max([abs(det_list[i][0][z] - det_list[j][0][z]) for z in range(2)]) < self.max_devloc:
                    det_list[i][0][0] = min(det_list[i][0][0], det_list[j][0][0])
                    det_list[i][0][1] = max(det_list[i][0][1], det_list[j][0][1])
                    det_list[i][0][2] = min(det_list[i][0][2], det_list[j][0][2])
                    det_list[i][0][3] = max(det_list[i][0][3], det_list[j][0][3])
                    det_list[i][3] += det_list[j][2] - det_list[j][1]
        return det_list

    def max_time(self, det_list):
        max_subvideo = 0
        for i in range(len(det_list)):
            if det_list[i][3] > det_list[max_subvideo][3]:
                if self.select_down:
                    max_subvideo = i
                elif det_list[i][0][0] > self.height / 2:
                    max_subvideo = i
        return det_list[max_subvideo][0], self.begin_t, self.end_t

    def detect(self):
        cap = cv2.VideoCapture(self.vd_path)
        # 检查视频是否成功打开
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()
        reader = easyocr.Reader(["ch_sim"])
        det_list = []
        sub_starttime = 0
        
        # 逐帧读取视频
        print(self.vd_path)
        print("检测中...")
        last_text = False
        while True:
            # 读取视频帧
            ret, frame = cap.read()
            if self.height == 0:
                self.height, self.width = frame.shape[:2]
                print(self.height, self.width)
            
            if self.begin_t == None and self.end_t == None:
                self.select_down = False
            if self.begin_t == None:
                self.begin_t = 0
            if self.end_t == None:
                # 读取帧率
                fps = cap.get(cv2.CAP_PROP_FPS)
                # 读取帧数
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # 计算视频时长
                self.end_t = frame_count / fps
            # 如果正确读取帧，ret为True
            if not ret:
                print("End")
                break
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if (
                timestamp >= self.begin_t
                and timestamp <= self.end_t
                and timestamp % self.det_time == 0
            ):  # timestamp >= begin_t and  timestamp<= end_t and
                result = reader.readtext(frame)
                last_text = (len(result) != 0) #有内容为True
                # 获取当前帧的时间戳（单位为秒）
                if self.print_time:
                    print(f"Timestamp: {timestamp} seconds")
                if self.print_res:
                    print(RED_FONT + "result" + END)
                    print(result)
                out_list = []
                if len(result) != 0:
                    result = result[::-1]
                    result_out = []
                    for i in range(len(result)):
                        if "int" in str(type(result[i][0][0][0])):
                            result_out = result[i]
                            break
                    if self.print_selectres:
                        print(RED_FONT + "result_out" + END)
                        print(result_out)
                    if len(result_out) != 0:  # (y1, x2, y2, x2)
                        out_list.append(result_out[0][0][1])
                        out_list.append(result_out[0][2][1])
                        out_list.append(result_out[0][0][0])
                        out_list.append(result_out[0][2][0])
                        out_list = [int(i) for i in out_list]
                if (
                    len(out_list) != 0
                    and len(det_list) != 0
                    and max([abs(out_list[i] - det_list[-1][0][i]) for i in range(2)]) < self.max_devloc
                ):
                    det_list[-1][0][0] = min(out_list[0], det_list[-1][0][0])
                    det_list[-1][0][1] = max(out_list[1], det_list[-1][0][1])
                    det_list[-1][0][2] = min(out_list[2], det_list[-1][0][2])
                    det_list[-1][0][3] = max(out_list[3], det_list[-1][0][3])
                    det_list[-1][2] = timestamp
                    sub_starttime = timestamp
                elif len(out_list) != 0:
                    sub_endtime = timestamp
                    det_list.append([out_list, sub_starttime, sub_endtime])
                    sub_starttime = timestamp
                elif last_text is False:
                    sub_starttime = timestamp
                
        if self.print_totaloutlist:
            print("det_list", det_list)
        det_list = self.del_repeat(det_list)
        if self.print_delrepeatoutlist:
            print(det_list)  # (y1, y2, x1, x2)
        det_list = [self.max_time(det_list)]
        percent_list=[det_list[0][0][0]/self.height,det_list[0][0][1]/self.height,det_list[0][0][2]/self.width,det_list[0][0][3]/self.width]
        print( self.vd_path ,"Subtitle position",det_list)
        print( self.vd_path ,"Subtitle precent",percent_list)
        # 释放视频对象
        cap.release()
        return det_list, percent_list
