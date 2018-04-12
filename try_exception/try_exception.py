#!/usr/bin/ecv python3
# -*- coding: UTF-8 -*-

try:
    fh = open("testfile", "w")
    fh.write("这是测试文件,写入正常")
except IOError:
    print("没有找到文件或者文件存在异常")
else:
    print("文件写入正常")
    fh.close()