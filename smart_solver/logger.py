import os
import logging
if not os.path.exists('logs'):
       os.mkdir('logs')
logging.basicConfig(filename='logs/log.log',
                    format="%(asctime)s -- %(pathname)s --%(filename)s-- %(module)s --\
             %(funcName)s -- %(lineno)d-- %(name)s -- %(levelname)s -- %(message)s",
                    level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s -- %(filename)s -- %(lineno)d -- %(name)s -- %(levelname)s -- %(message)s")
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
