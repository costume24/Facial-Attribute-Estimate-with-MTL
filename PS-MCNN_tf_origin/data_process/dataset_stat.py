from __future__ import absolute_import
from __future__ import division

import shutil
import os

output_path = "/media/xuke/Files/Final/DataSet"
CelebA_Attr_file = "/media/xuke/SoftWare/BaiduNetdiskDownload/CelebA/Anno/list_attr_celeba.txt"
'''
统计各属性的正负样本比例
'''


def main():

    result_txt = open(os.path.join(output_path, "result_txt.txt"), "w")
    res = [0] * 40
    with open(CelebA_Attr_file, "r") as Attr_file:
        Attr_info = Attr_file.readlines()
        Total = int(Attr_info[0])
        Attr_name = Attr_info[1].split()
        Attr_info = Attr_info[2:]
        index = 0
        for line in Attr_info:
            Attr_type = 1
            index += 1
            info = line.split()
            filename = info[0]
            while Attr_type < 41:
                if int(info[Attr_type]) == 1:
                    res[Attr_type - 1] += 1
                Attr_type += 1

        for i in range(40):
            result_txt.writelines("%s                Pos: %d  Neg: %d \n" %
                                  (Attr_name[i], res[i], Total - res[i]))

    result_txt.close()


if __name__ == "__main__":
    main()