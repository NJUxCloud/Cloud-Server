"""
add by wsw
    所有的预处理函数都在这里添加
    views 不用修改
    init是所有级联函数的最高层
    如果要写一个预处理函数:
    首先在其上一层的return中,添加预处理函数名和预处理中文显示名
    然后填写参数,每个参数要加入annotation(格式仿照4个具体resize
    格式为:("该参数在界面的中文显示","该参数的类型 e.g. str或float",该参数的下限,该函参数上限)
    后两个为界面提供输入的范围

    在扩号后面 接入 -> True/False
    标明这个函数是否是最终函数
"""

"""
初始化
"""
def init()->False:
    return {
       'functions':  [
            {
                'func': 'resize',
                'name': '图像放缩'
            },

        ]
    }
"""
图像大小重构
"""
def resize()-> False:
    return {
       'functions': [
            {
                'func': 'resize_nearest',
                'name': '邻域法'
            },
            {
                'func': 'resize_bicubic',
                'name': '三次插值法'
            },
            {
                'func': 'resize_bilinear',
                'name': '双线性插值法'
            },
            {
                'func': 'resize_antialias',
                'name': '平滑法'
            },

        ]
    }

"""
邻域法
"""
def resize_nearest(dir:("路径", "str", None, None), new_x:("长度", "float", 1, 200), new_y:("宽度", "float", 1, 200)) -> True:
    print("you are resizing image in " + dir + ":  x: " + str(new_x) + " ,y: " + str(new_y) + "  by nearest")

"""
三次插值法
"""
def resize_bicubic(dir:("路径", "str", None, None), new_x:("长度", "float", 1, 200), new_y:("宽度", "float", 1, 200)) -> True:
    print("you are resizing image in " + dir + ":  x: " + str(new_x) + " ,y: " + str(new_y) + "  by bicubic")

"""
双线性插值法
"""
def resize_bilinear(dir:("路径", "str", None, None), new_x:("长度", "float", 1, 200), new_y:("宽度", "float", 1, 200)) -> True:
    print("you are resizing image in " + dir + ":  x: " + str(new_x) + " ,y: " + str(new_y) + "  by bilinear")

"""
平滑法
"""
def resize_antialias(dir:("路径", "str", None, None), new_x:("长度", "float", 1, 200), new_y:("宽度", "float", 1, 200)) -> True:
    print("you are resizing image in " + dir + ":  x: " + str(new_x) + " ,y: " + str(new_y) + "  by ntialias")