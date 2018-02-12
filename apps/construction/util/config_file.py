from configparser import ConfigParser
import traceback


class Config:
    def __init__(self, init_file_path):
        self.path = init_file_path
        self.cf = ConfigParser()
        self.cf.read(self.path)

    def get(self, section, key):
        try:
            result = self.cf.get(section, key)
        except:
            result = ""
        return result

    def set(self, section, key, value):
        try:
            if not self.cf.has_section(section):
                self.cf.add_section(section)

            self.cf.set(section, key, value)
            self.cf.write(open(self.path, 'w'))
        except:
            # print(traceback.print_exc())
            return False
        return True


# config_file_path = '/Users/keenan/Downloads/test.ini'
# config = Config(init_file_path=config_file_path)
#
# print(config.set(section='db', key='username', value='root'))
# config.set(section='db', key='password', value='123456')
# config.set(section='db', key='aaa', value='1233')
#
# config.set(section='boot', key='bcns', value='123')
