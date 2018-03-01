# coding: utf-8

import paramiko
import re
from time import sleep
import traceback
from stat import S_ISDIR
import os


# 定义一个类，表示一台远端linux主机
from CloudServer import global_settings


class Linux(object):
    # 通过IP, 用户名，密码，超时时间初始化一个远程Linux主机
    # def __init__(self, ip, username, password, timeout=30):
    #     self.ip = ip
    #     self.username = username
    #     self.password = password
    #     self.timeout = timeout
    #     # transport和chanel
    #     self.t = ''
    #     self.chan = ''
    #     # 链接失败的重试次数
    #     self.try_times = 3

    def __init__(self, timeout=30):
        '''
        此处先将远程主机的用户名密码定死
        '''
        self.ip = '119.23.51.139'
        self.username = 'root'
        self.password = 'NJUCloud145'
        self.timeout = timeout
        # transport和chanel
        self.t = ''
        self.chan = ''
        # 链接失败的重试次数
        self.try_times = 3

    def connect(self):
        """调用该方法连接远程主机"""
        while True:
            # 连接过程中可能会抛出异常，比如网络不通、链接超时
            try:
                # 设置ssh连接的远程主机地址和端口
                self.t = paramiko.Transport(sock=(self.ip, 22))
                # 设置登录名和密码
                self.t.connect(username=self.username, password=self.password)
                # 连接成功后打开一个channel
                self.chan = self.t.open_session()
                # 设置会话超时时间
                self.chan.settimeout(self.timeout)
                # 打开远程的terminal
                self.chan.get_pty()
                # 激活terminal
                self.chan.invoke_shell()
                # 设置sftp
                self.sftp = paramiko.SFTPClient.from_transport(self.t)
                # 如果没有抛出异常说明连接成功，直接返回
                print(u'连接%s成功' % self.ip)
                # 接收到的网络数据解码为str
                print(self.chan.recv(65535).decode('utf-8'))
                return
            # 这里不对可能的异常如socket.error, socket.timeout细化，直接一网打尽
            except Exception as e1:
                if self.try_times != 0:
                    print(u'连接%s失败，进行重试' % self.ip)
                    self.try_times -= 1
                else:
                    print(u'重试3次失败，结束程序')
                    exit(1)

    def close(self):
        """
        断开连接
        :return:
        """
        self.chan.close()
        self.t.close()

    def send(self, cmd):
        """
        发送要执行的命令
        :param cmd:
        :return:
        """
        cmd += '\r'
        # 通过命令执行提示符来判断命令是否执行完成
        p = re.compile(r'#')

        result = ''
        # 发送要执行的命令
        self.chan.send(cmd)
        # 回显很长的命令可能执行较久，通过循环分批次取回回显
        while True:
            sleep(0.5)
            ret = self.chan.recv(65535)
            ret = ret.decode('utf-8')
            result += ret
            if p.search(ret):
                print(result)
                return result

    def sftp_upload_file(self,  dir_path, file_path,need_unzip):
        """
        上传文件
        """
        try:
            cmd = 'test -d ./' + dir_path + ' || mkdir -p ' + dir_path
            self.send(cmd)
            self.sftp.put(global_settings.LOCAL_STORAGE_PATH + file_path, file_path)
            if(need_unzip):
                self.unzip_file(file_path)
        except Exception as e:
            print(e)

    def unzip_file(self, file_path):
        """
        解压文件 unzip
        :param file_path: 文件目录
        :return:
        """
        unzip_dir_path = file_path.split('.')[0]
        try:
            cmd = 'unzip ./' + file_path + ' -d ' + unzip_dir_path + ' && rm -rf ' + unzip_dir_path + '__MACOSX'
            self.send(cmd)
        except Exception as e:
            print(e)

    def download(self, remote_path, local_path):
        """
        递归下载远程服务器的整个目录或文件，并保持目录结构
        :param remote_path: 远程文件或目录名称（绝对路径）
        :param local_path: 本地文件或目录名称（绝对路径）
        :return:
        """
        if not self.is_dir(remote_path):
            if not os.path.exists(local_path):
                path = os.path.split(local_path)[0]
                os.makedirs(path)
            self.sftp.get(remotepath=remote_path, localpath=local_path)
            return

        item_list = self.sftp.listdir(remote_path)
        dest = str(local_path)
        print(dest)
        if not os.path.isdir(dest):
            os.mkdir(path=dest)

        for item in item_list:
            item = str(item)
            if self.is_dir(path=remote_path + '/' + item):
                self.download(remote_path=remote_path + '/' + item, local_path=dest + '/' + item)
            else:
                self.sftp.get(remotepath=remote_path + '/' + item, localpath=dest + '/' + item)

    def is_dir(self, path):
        """
        判断远程服务器的一个路径是否是目录(True)或文件(False)
        :param path:
        :return:
        """
        try:
            return S_ISDIR(self.sftp.stat(path).st_mode)
        except IOError:
            return False


# host = Linux()
# host.connect()
# host.download(remote_path='NJUCloud/1/data/doc/url_20180107161753',
#               local_path='/Users/keenan/Downloads/url_20180107161753')
# host.close()
