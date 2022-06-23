'''
Author: whr1241 2735535199@qq.com
Date: 2022-04-19 14:34:56
LastEditors: whr1241 2735535199@qq.com
LastEditTime: 2022-06-16 22:15:57
FilePath: \InfraredPPG1.1\0test.py
Description: 性能分析
'''
import cProfile
import Green
cProfile.run(filename = 'Green.py', sort=-1)