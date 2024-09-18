import ftplib
from loguru import logger
import threading

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

    def upload_file(self, img, ftpurl):
        """上传现场图像文件
        Args:
            local_file (str): 本地文件路径.
            remote_file (str): 上传到FTP服务器的路径
        """
        if len(img) == len(ftpurl):
            for local_file, remote_file in zip(img, ftpurl):
                with open(local_file, 'rb') as f:
                    self.ftp.storbinary(f"STOR {remote_file}", f)
                    logger.info(f"Uploaded file from {local_file} to {remote_file}")

    def download_file(self, local_file_path, remote_file_path):
        """下载配置文件
        Args:
            local_file_path (str): 本地文件路径.
            remote_file_path (str): FTP服务器上待下载的文件路径.
        """
        with open(local_file_path, 'wb') as local_file:
            self.ftp.retrbinary(f"RETR {remote_file_path}", local_file.write)
            logger.info(f"Download file from {remote_file_path} to {local_file_path}")

    def ftp_close(self):
        self.ftp.quit()