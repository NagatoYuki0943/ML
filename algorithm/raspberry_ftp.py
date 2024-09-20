import ftplib
from loguru import logger
import threading
import os

class RaspberryFTP:
    def __init__(
        self,
        ip: str = "localhost",
        port: int = 21,
        username: str = "admin",
        password: str = "123456",
    ):
        """
        Args:
            ip (str): FTP服务器IP地址. Defaults to "localhost".
            port (int): FTP服务器端口. Defaults to 21.
            username (str): 连接FTP服务器用户名. Defaults to "admin".
            password (str): 连接FTP服务器密码.
        """
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password

        self.ftp = ftplib.FTP()

    def ftp_connect(self):
        try:
            self.ftp.connect(self.ip, self.port)
            self.ftp.login(self.username, self.password)
            logger.info(f"FTP server connected")
        except ftplib.Error as e:
            logger.error(f"FTP server unreachable : {e}")
            raise

    def upload_file(self, local_file_path, local_file_name, ftpurl):
        """上传现场文件
        Args:
            local_file_path (list[str] | str): 本地文件路径.
            local_file_name (list[str] | str): 上传的文件名
            ftpurl (str): 上传到FTP服务器的路径.
        """
        # 单个字符串转换为列表
        try:
            if isinstance(local_file_name, str):
                local_file_name = [local_file_name]
            if isinstance(local_file_path, str):
                local_file_path = [local_file_path]
            # 检查每个本地文件是否存在
            for file_path in local_file_path:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Local file does not exist: {file_path}")
            # 创建子目录
            self.ftp.mkd(ftpurl)
            # 本地路径和名字绑定，上传文件
            for file_path, file_name in zip(local_file_path, local_file_name):
                remote_file = f"{ftpurl}/{file_name}"
                with open(file_path, 'rb') as f:
                    self.ftp.storbinary(f"STOR {remote_file}", f)
                    logger.info(f"Uploaded file from {file_path} to {remote_file}")
        except FileNotFoundError as e:
            logger.error(e)
            raise
        except Exception as e:
            self.ftp.rmd(ftpurl)
            logger.error(f"FTP uploads error:{e}")
            raise

    def download_file(self, local_file_path, remote_file_name, ftpurl):
        """下载配置文件
        Args:
            local_file_path (str): 本地文件路径.
            remote_file_name (str): FTP服务器上待下载的文件名.
            ftpurl (str): FTP服务器上的下载路径
        """
        try:
            remote_file = f"{ftpurl}/{remote_file_name}"
            with open(local_file_path, 'wb') as f:
                self.ftp.retrbinary(f"RETR {remote_file}", f.write)
                logger.info(f"Downloaded {local_file_path} from {remote_file}")
        except ftplib.error_perm as e:
            logger.error(f"FTP downloads error:{e}")
            raise

    def check_and_connect(self):
        """检查连接状态，断开时重连"""
        try:
            self.ftp.voidcmd('NOOP')
        except ftplib.Error as e:
            self.ftp_connect()

    def ftp_close(self):
        self.ftp.quit()