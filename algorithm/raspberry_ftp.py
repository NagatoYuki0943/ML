import ftplib
from loguru import logger
import os
import time

class RaspberryFTP:
    def __init__(
        self,
        ip: str = "120.79.11.147",
        port: int = 21,
        username: str = "vision",
        password: str = "GLspepec123",
        max_retries: int = 3,
        delay: int = 1,
        pasv_mode: bool = False
    ):
        """
        Args:
            ip (str): FTP服务器IP地址. Defaults to "localhost".
            port (int): FTP服务器端口. Defaults to 21.
            username (str): 连接FTP服务器用户名. Defaults to "admin".
            password (str): 连接FTP服务器密码.
            max_retries (int): 最大重试次数. Defaults to 3.
            delay (int): 重试间隔时间(秒). Defaults to 1.
            pasv_mode (bool): 使用主动模式连接服务器. Defaults to False.
        """
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        self.max_retries = max_retries
        self.delay = delay
        self.pasv_mode = pasv_mode
        self.ftp = ftplib.FTP()


    def ftp_connect(self):
        try:
            self.ftp.connect(self.ip, self.port, timeout=5)
            self.ftp.login(self.username, self.password)
            self.ftp.set_pasv(self.pasv_mode)
            logger.info("FTP server connected")
        except Exception as e:
            logger.error(f"FTP server unreachable : {e}")
            raise

    def upload_file(self, local_file_path, local_file_name, ftpurl):
        """上传现场文件
        Args:
            local_file_path (list[str] | str): 本地文件路径.
            local_file_name (list[str] | str): 上传的文件名
            ftpurl (str): 上传到FTP服务器的路径.
        """
        delay_uploads = self.delay
        # 单个字符串转换为列表
        if isinstance(local_file_name, str):
            local_file_name = [local_file_name]
        if isinstance(local_file_path, str):
            local_file_path = [local_file_path]
        # 检查每个本地文件是否存在
        for file_path in local_file_path:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Local file does not exist: {file_path}")
        for attempt in range(self.max_retries):
            try:
                self.ftp_connect()
                # 检查并创建子目录
                try:
                    self.ftp.cwd(ftpurl)
                except ftplib.error_perm:
                    self.ftp.mkd(ftpurl)
                self.ftp.cwd(ftpurl)
                # 绑定路径与文件名进行上传
                for file_path, file_name in zip(local_file_path, local_file_name):
                    with open(file_path, 'rb') as f:
                        self.ftp.storbinary(f"STOR {file_name}", f)
                    logger.info(f"Uploaded file from {file_path} to FTP server")
                self.ftp_close()
                break
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(e)
                    logger.warning(f"Uploads failed, retrying in {delay_uploads} seconds...(attempt {attempt + 1} / {self.max_retries})")
                    delay_uploads *= 2
                    time.sleep(delay_uploads)
                else:
                    logger.error(f"FTP uploads error:{e}")
                    self.ftp.rmd(ftpurl)
                    raise

    def download_file(self, local_file_path, remote_file_name, ftpurl):
        """下载配置文件
        Args:
            local_file_path (str): 本地文件路径.
            remote_file_name (str): FTP服务器上待下载的文件名.
            ftpurl (str): FTP服务器上的下载路径
        """
        delay_downloads = self.delay
        for attempt in range(self.max_retries):
            try:
                self.ftp_connect()
                remote_file = f"{ftpurl}/{remote_file_name}"
                with open(local_file_path, 'wb') as f:
                    self.ftp.retrbinary(f"RETR {remote_file}", f.write)
                self.ftp_close()
                logger.info(f"Downloaded {local_file_path} from {remote_file}")
                break
            except ftplib.Error as e:
                if attempt < self.max_retries:
                    logger.warning(f"Downloads failed, retrying in {delay_downloads} seconds...(attempt {attempt + 1} / {self.max_retries})")
                    delay_downloads *= 2
                    time.sleep(delay_downloads)
                else:
                    logger.error(f"FTP downloads error:{e}")
                    raise

    def ftp_close(self):
        self.ftp.quit()
        logger.info("FTP disconnected")