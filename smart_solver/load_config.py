import sys
from os import path
from configparser import ConfigParser

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

config = ConfigParser()
config.read('config/config.ini')

database_type = config.get('database','database_type')
user = config.get('database','user')
password = config.get('database','password')
host = config.get('database','host')
database_name = config.get('database','database_name')

conn = database_type+'://'+user+':'+password+'@'+host+'/'+database_name
