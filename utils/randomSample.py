#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import random
import shutil


def moveFile(src, dst, rate):
    pathDir = [img for img in os.listdir(src)]
    filenumber = len(pathDir)  # the number of images
    sample = random.sample(pathDir, int(filenumber * rate))  # sample的文件路径
    for name in sample:
        print(name)
        # move file
        # shutil.move(src + name, dst + name)
        # copy file
        shutil.copyfile(os.path.join(src, name), os.path.join(dst, name))
    return


if __name__ == '__main__':
    src = r''
    dst = r''
    moveFile(src, dst, 0.5)
