import argparse
import importlib
import multiprocessing
import os
import pathlib
import platform
import sys
import tempfile
import time
import shutil
import subprocess

import cv2
import easyocr
import tqdm

import cfg
from backend.inpaint.sttn_inpaint import STTNVideoInpaint
from backend.tools.inpaint_tools import create_mask

# 定义一个红色字体的ANSI转义代码
# RED_FONT = "\033[31m"
# 重置所有文本属性的ANSI转义代码
# END = "\033[0m"


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
            # print_res=False,
            # print_time=False,
            # print_selectres=False,
            # print_totaloutlist=False,
            # use_deltime=False,
            # print_delrepeatoutlist=False,
    ):
        self.vd_path = vd_path
        self.video_cap = cv2.VideoCapture(self.vd_path)
        if not self.video_cap.isOpened():
            print(f"Could not open video {self.vd_path}")
            exit(1)
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frame_height = self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.begin_t = begin_t if begin_t else 0
        self.end_t = end_t if end_t else self.frame_count / self.fps
        self.det_time = det_time
        self.set_dettime = set_dettime
        self.max_devloc = max_devloc
        # self.print_res = print_res
        # self.print_time = print_time
        # self.print_selectres = print_selectres
        # self.print_totaloutlist = print_totaloutlist
        # self.use_deltime = use_deltime
        # self.print_delrepeatoutlist = print_delrepeatoutlist
        self.select_down = True

    def del_repeat(self, det_list):
        for i in range(len(det_list)):
            det_list[i].append(det_list[i][2] - det_list[i][1])
            for j in range(i + 1, len(det_list)):
                if max([abs(det_list[i][0][z] - det_list[j][0][z]) for z in range(2)]) < self.max_devloc:
                    det_list[i][0][0] = min(det_list[i][0][0], det_list[j][0][0])
                    det_list[i][0][1] = max(det_list[i][0][1], det_list[j][0][1])
                    det_list[i][0][2] = min(det_list[i][0][2], det_list[j][0][2])
                    det_list[i][0][3] = max(det_list[i][0][3], det_list[j][0][3])
                    det_list[i][3] += det_list[j][2] - det_list[j][1]
        return det_list

    def max_time(self, det_list):
        max_sub_video = 0
        for i in range(len(det_list)):
            if det_list[i][3] > det_list[max_sub_video][3]:
                if self.select_down:
                    max_sub_video = i
                elif det_list[i][0][0] > self.frame_height / 2:
                    max_sub_video = i
        return det_list[max_sub_video][0], self.begin_t, self.end_t

    def run(self):
        reader = easyocr.Reader(
            cfg.ocr_lang_list,
            model_storage_directory=cfg.OCR_MODEL_DIR,
            user_network_directory=cfg.OCR_MODEL_DIR,
            # TODO: recognizer=False
            recognizer=True
        )
        det_list = []
        sub_start_t = 0
        last_text = False
        while True:
            # 读取视频帧
            ret, frame = self.video_cap.read()
            if not ret:
                print(f"Frame cannot be read {self.vd_path}")
                break
            if self.begin_t is None and self.end_t is None:
                self.select_down = False
            timestamp = self.video_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if self.begin_t < timestamp < self.end_t and timestamp % self.det_time == 0:
                result,_ = reader.detect(frame)
                if not result:
                    continue
                result = result[0]
                last_text = (len(result) != 0) #有内容为True
                out_list=[]
                if len(result) != 0:
                    result = result[::-1]
                    out_list=[result[0][2],result[0][3],result[0][0],result[0][1]]
                    out_list = [max(0,int(i)) for i in out_list]
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
                    sub_start_t = timestamp
                elif len(out_list) != 0:
                    sub_end_t = timestamp
                    det_list.append([out_list, sub_start_t, sub_end_t])
                    sub_start_t = timestamp
                elif last_text is False:
                    sub_start_t = timestamp
        # if self.print_totaloutlist:
        #     print("det_list", det_list)
        det_list = self.del_repeat(det_list)
        # if self.print_delrepeatoutlist:
        #     print(det_list)  # (y1, y2, x1, x2)
        det_list = [self.max_time(det_list)]
        print(det_list)
        # 释放视频对象
        self.video_cap.release()
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        return det_list[0][0]


class SubtitleRemover:

    def __init__(
            self, vd_path, sub_area=None, gui_mode=False, start_t=None, end_t=None
    ):
        importlib.reload(cfg)
        self.sub_area = sub_area
        self.gui_mode = gui_mode
        self.is_picture = False
        self.video_path = vd_path
        self.video_cap = cv2.VideoCapture(vd_path)
        self.vd_name = pathlib.Path(self.video_path).stem
        self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        self.start_frame = 0 if start_t is None else int(start_t * self.fps)
        self.end_frame = self.frame_count if end_t is None else int(end_t * self.fps)
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.size = (self.frame_width, self.frame_height)
        self.mask_size = (self.frame_height, self.frame_width)
        self.video_temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.video_writer = cv2.VideoWriter(
            self.video_temp_file.name, cv2.VideoWriter.fourcc(*"mp4v"),
            self.fps, self.size
        )
        self.video_out_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(self.video_path))),
            "videos_no_sub"
        )
        if not os.path.exists(self.video_out_dir):
            os.makedirs(self.video_out_dir)
        self.video_out_name = os.path.join(
            self.video_out_dir, f"{self.vd_name}_no_sub.mp4"
        )
        self.video_inpaint = None
        self.lama_inpaint = None
        self.ext = os.path.splitext(vd_path)[-1]
        self.progress_total = 0
        self.progress_remover = 0
        self.is_finished = False
        self.preview_frame = None
        self.is_successful_merged = False

    def update_progress(self, tbar, increment):
        tbar.update(increment)
        current_percentage = (tbar.n / tbar.total) * 100
        self.progress_remover = int(current_percentage) // 2
        self.progress_total = 50 + self.progress_remover

    def sttn_mode_with_no_detection(self, tbar):
        if self.sub_area is not None:
            ymin, ymax, xmin, xmax = self.sub_area
        else:
            print(
                "[Info] No subtitle area has been set. "
                "Video will be processed in full screen. "
                "As a result, the final outcome might be suboptimal."
            )
            ymin, ymax, xmin, xmax = 0, self.frame_height, 0, self.frame_width
        mask_area_coordinates = [(xmin, xmax, ymin, ymax)]
        mask = create_mask(self.mask_size, mask_area_coordinates)
        sttn_video_inpaint = STTNVideoInpaint(self.video_path)
        sttn_video_inpaint(
            input_mask=mask, input_sub_remover=self, tbar=tbar,
            start_frame=self.start_frame, end_frame=self.end_frame
        )

    def sttn_mode(self, tbar):
        if cfg.STTN_SKIP_DETECTION:
            self.sttn_mode_with_no_detection(tbar)

    def run(self):
        start_time = time.time()
        self.progress_total = 0
        tbar = tqdm.tqdm(
            total=int(self.frame_count),
            unit="frame",
            position=0,
            file=sys.__stdout__,
            desc=f"Subtitle Removing {self.vd_name}"
        )
        print()
        if cfg.MODE == cfg.InpaintMode.STTN:
            self.sttn_mode(tbar)
        self.video_cap.release()
        self.video_writer.release()
        print()
        if not self.is_picture:
            self.merge_audio_to_video()
            print(f"[Finished] Video generated at: {self.video_out_name}")
        else:
            print(f"[Finished] Picture generated at: {self.video_out_name}")
        end_time = time.time()
        time_cost = end_time - start_time
        print(f"Time cost {self.vd_name}: {nice_time_cost(time_cost)}")
        self.is_finished = True
        self.progress_total = 100
        if os.path.exists(self.video_temp_file.name):
            try:
                os.remove(self.video_temp_file.name)
            except OSError:
                if platform.system() in ["Windows"]:
                    pass
                else:
                    print(f"Failed to delete temp file {self.video_temp_file.name}")

    def merge_audio_to_video(self):
        temp = tempfile.NamedTemporaryFile(suffix='.aac', delete=False)
        audio_extract_command = [
            cfg.FFMPEG_PATH,
            "-y", "-i", self.video_path,
            "-acodec", "copy", "-vn", "-loglevel", "error", temp.name
        ]
        use_shell = True if os.name == "nt" else False
        try:
            subprocess.check_output(
                audio_extract_command, stdin=open(os.devnull), shell=use_shell
            )
        except subprocess.CalledProcessError:
            print("Fail to extract audio")
            return
        else:
            if os.path.exists(self.video_temp_file.name):
                audio_merge_command = [
                    cfg.FFMPEG_PATH,
                    "-y", "-i", self.video_temp_file.name, "-i", temp.name,
                    "-vcodec", "libx264" if cfg.USE_H264 else "copy",
                    "-acodec", "copy", "-loglevel", "error", self.video_out_name
                ]
                try:
                    subprocess.check_output(
                        audio_merge_command, stdin=open(os.devnull), shell=use_shell
                    )
                except subprocess.CalledProcessError:
                    print("Fail to merge audio")
                    return
            if os.path.exists(temp.name):
                try:
                    os.remove(temp.name)
                except OSError:
                    if platform.system() in ["Windows"]:
                        pass
                    else:
                        print(f"Failed to delete temp file {temp.name}")
            self.is_successful_merged = True
        finally:
            temp.close()
            if not self.is_successful_merged:
                try:
                    shutil.copy2(self.video_temp_file.name, self.video_out_name)
                except IOError as e:
                    print(f"Unable to copy file. {e}")
            self.video_temp_file.close()


def nice_time_cost(time_cost):
    hours, minutes = divmod(time_cost, 3600)
    minutes, seconds = divmod(minutes, 60)
    if hours:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes:
        return f"{int(minutes)}m {int(seconds)}s"
    elif seconds:
        return f"{int(seconds)}s"


def process_files_in_directory(
        directory, files, sub_area=None, start_t=None, end_t=None
):
    need_detection = False
    for file in files:
        file_path = os.path.join(directory, file)
        if sub_area is None or need_detection:
            need_detection = True
            detector = SubtitleDetector(file_path, begin_t=start_t, end_t=end_t)
            sub_area = detector.run()
        remover = SubtitleRemover(
            file_path, sub_area=sub_area, start_t=start_t, end_t=end_t
        )
        remover.run()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser(description="video subscript remover")
    parser.add_argument(
        "--dir", default=cfg.INPUT_DIR,
        help="video absolute directory path"
    )
    parser.add_argument(
        "--area",
        help="subtitle area (y1, y2, x1, x2)"
    )
    parser.add_argument(
        "--start",
        help="start time"
    )
    parser.add_argument(
        "--end",
        help="end time"
    )
    args = vars(parser.parse_args())
    video_directory_path = args["dir"]
    if not os.path.exists(video_directory_path):
        print(f"Directory path does not exist: {video_directory_path}")
        sys.exit()
    subtitle_area = args["area"]
    if isinstance(subtitle_area, str):
        subtitle_area = eval(subtitle_area)
        if not isinstance(subtitle_area, tuple) and len(subtitle_area) != 4:
            print(f"Subtitle area not correct: {subtitle_area}")
            sys.exit(0)
    subtitle_start_time = args["start"]
    if subtitle_start_time and isinstance(subtitle_start_time, str):
        subtitle_start_time = eval(subtitle_start_time)
        if not isinstance(subtitle_start_time, int):
            print(f"Subtitle start time not correct: {subtitle_start_time}")
            sys.exit(0)
    subtitle_end_time = args["end"]
    if subtitle_end_time and isinstance(subtitle_end_time, str):
        subtitle_end_time = eval(subtitle_end_time)
        if not isinstance(subtitle_end_time, int):
            print(f"Subtitle end time not correct: {subtitle_end_time}")
            sys.exit(0)
    video_paths = os.listdir(video_directory_path)
    num_videos = len(video_paths)
    num_processes = min(
        num_videos, multiprocessing.cpu_count(), cfg.MAX_PROCESSES
    )
    chunk_size = num_videos // num_processes
    remainder = num_videos % num_processes
    chunks = []
    chunk_start = 0
    for process_no in range(num_processes):
        if process_no < remainder:
            chunk_end = chunk_start + chunk_size + 1
        else:
            chunk_end = chunk_start + chunk_size
        chunks.append(video_paths[chunk_start:chunk_end])
        chunk_start = chunk_end
    all_start_time = time.time()
    pool = multiprocessing.Pool(processes=num_processes)
    pool.starmap(
        process_files_in_directory,
        [
            (
                video_directory_path, chunk, subtitle_area,
                subtitle_start_time, subtitle_end_time
            )
            for chunk in chunks
        ]
    )
    pool.close()
    pool.join()
    all_end_time = time.time()
    all_time_cost = all_end_time - all_start_time
    print(
        f"Subtitles of all {num_videos} videos removed "
        f"within {nice_time_cost(all_time_cost)}"
    )
