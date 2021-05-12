# encoding: utf-8

"""
Downloading Data files and organize data file structures

"""

import urllib.request
import shutil
import os

# URLs for the zip files
def download_data():
    links = [
        'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
        'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
        'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
        'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
        'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
        'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
        'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
        'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
        'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
        'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
        'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
        'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
    ]

    for idx, link in enumerate(links):
        fn = 'images_%02d.tar.gz' % (idx+1)
        print('downloading'+fn+'...')
        urllib.request.urlretrieve(link, fn)  # download the zip file

    print("Download complete. Please check the checksums")


def modify_data_list(list_type):
    IMAGE_LIST = f"/gdrive/My Drive/CS598-Project-Data/labels/{list_type}_list.txt"
    new_list = open(f"new_{list_type}_list.txt", 'w')
    for i in range(1, 13):
    source_dir = f"/gdrive/My Drive/CS598-Project-Data/images_{i:02d}/images"
    print(source_dir)
    
    file_names = os.listdir(source_dir)
    print(source_dir, "Num Files:", len(file_names))      

    num = 0
    with open(IMAGE_LIST, "r") as f:
        for line in f:
        items = line.split()
        image_name= items[0]
        if image_name in file_names:
            num = num + 1
            items.insert(0, f"images_{i:02d}")
            newline = " ".join(items)
            new_list.write(newline + "\n")

    print(f"Total {list_type} images in {i:02d}:", num)

    new_list.close()

if __name__ == __main__:
    download_data()
    modify_data_list("train")
    modify_data_list("val")
    modify_data_list("test")