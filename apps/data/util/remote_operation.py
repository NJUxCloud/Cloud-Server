# coding: utf-8

import paramiko
import re
from time import sleep

# 定义一个类，表示一台远端linux主机
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
        self.password = 'NJUCloud017'
        self.timeout = timeout
        # transport和chanel
        self.t = ''
        self.chan = ''
        # 链接失败的重试次数
        self.try_times = 3

    # 调用该方法连接远程主机
    def connect(self):
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
                print (u'连接%s成功' % self.ip)
                # 接收到的网络数据解码为str
                print (self.chan.recv(65535).decode('utf-8'))
                return
            # 这里不对可能的异常如socket.error, socket.timeout细化，直接一网打尽
            except Exception as e1:
                if self.try_times != 0:
                    print (u'连接%s失败，进行重试' %self.ip)
                    self.try_times -= 1
                else:
                    print (u'重试3次失败，结束程序')
                    exit(1)

    # 断开连接
    def close(self):
        self.chan.close()
        self.t.close()

    # 发送要执行的命令
    def send(self, cmd):
        print('send1')
        cmd += '\r'
        # 通过命令执行提示符来判断命令是否执行完成
        p = re.compile(r'#')

        result = ''
        # 发送要执行的命令
        self.chan.send(cmd)
        # 回显很长的命令可能执行较久，通过循环分批次取回回显
        while True:
            print('send2')
            sleep(0.5)
            ret = self.chan.recv(65535)
            ret = ret.decode('utf-8')
            result += ret
            if p.search(ret):
                print( result)
                return result

    # 上传文件
    def sftp_upload_file(self,file, dir_path,file_path):
        try:
            cmd='test -d ./'+dir_path+' || mkdir -p '+dir_path
            self.send(cmd)
            self.sftp.putfo(file,file_path)
        except Exception as e:
            print (e)

# host = Linux()
# host.connect()
# host.send('ls -l')
# host.close()
