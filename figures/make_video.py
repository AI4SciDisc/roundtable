#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF转MP4视频转换器
将PDF文件的每一页转换为视频帧，生成MP4文件
"""

import os
import sys
import io  # ✅ 提前导入
from pathlib import Path
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm  # ✅ 新增

class PDFToMP4Converter:
    def __init__(self, pdf_path, output_path=None, duration_per_page=3, fps=30):
        """
        初始化转换器
        """
        self.pdf_path = Path(pdf_path)
        self.duration_per_page = duration_per_page
        self.fps = fps

        if output_path:
            self.output_path = Path(output_path)
        else:
            # ✅ 默认输出到当前目录
            self.output_path = Path.cwd() / (self.pdf_path.stem + '.mp4')

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {self.pdf_path}")

    def pdf_to_images(self, dpi=150):
        """
        将PDF页面转换为图像
        """
        print(f"正在读取PDF文件: {self.pdf_path}")
        doc = fitz.open(str(self.pdf_path))
        images = []

        print(f"PDF共有 {len(doc)} 页")
        for page_num in tqdm(range(len(doc)), desc="PDF转图片进度"):
            page = doc.load_page(page_num)
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
            pix = None

        doc.close()
        return images

    def images_to_video(self, images):
        """
        将图像列表转换为MP4视频
        """
        if not images:
            raise ValueError("没有图像可以转换为视频")

        print(f"正在创建视频: {self.output_path}")
        first_img = images[0]
        width, height = first_img.size

        if width % 2 == 1:
            width -= 1
        if height % 2 == 1:
            height -= 1

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (width, height))
        if not video_writer.isOpened():
            raise RuntimeError("无法创建视频文件")

        frames_per_page = int(self.fps * self.duration_per_page)

        # ✅ tqdm 包装总帧数
        total_frames = len(images) * frames_per_page
        with tqdm(total=total_frames, desc="写入视频进度") as pbar:
            for img in images:
                img_resized = img.resize((width, height), Image.Resampling.LANCZOS)
                img_array = np.array(img_resized)
                if len(img_array.shape) == 3:
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

                for _ in range(frames_per_page):
                    video_writer.write(img_bgr)
                    pbar.update(1)

        video_writer.release()
        print(f"视频创建完成: {self.output_path}")

    def convert(self, dpi=150):
        try:
            images = self.pdf_to_images(dpi)
            self.images_to_video(images)
            print("转换完成！")
        except Exception as e:
            print(f"转换失败: {e}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='将PDF文件转换为MP4视频')
    parser.add_argument('pdf_file', help='输入的PDF文件路径')
    parser.add_argument('-o', '--output', help='输出的MP4文件路径（可选，默认当前目录）')
    parser.add_argument('-d', '--duration', type=float, default=0.5,
                       help='每页显示时长（秒），默认3秒')
    parser.add_argument('--fps', type=int, default=30,
                       help='视频帧率，默认30fps')
    parser.add_argument('--dpi', type=int, default=300,
                       help='图像分辨率DPI，默认300')
    args = parser.parse_args()

    converter = PDFToMP4Converter(
        pdf_path=args.pdf_file,
        output_path=args.output,
        duration_per_page=args.duration,
        fps=args.fps
    )
    converter.convert(dpi=args.dpi)


if __name__ == "__main__":
    main()
