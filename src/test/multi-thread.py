import logging
import threading
import time

logging.basicConfig(level=logging.DEBUG, format='%(threadName)s: %(message)s')

def worker1():
    # thread の名前を取得
    logging.debug('start')
    time.sleep(5)
    logging.debug('end')

def worker2():
    logging.debug('start')
    time.sleep(5)
    logging.debug('end')

if __name__ == '__main__':
    # スレッドに workder1 関数を渡す
    t1 = threading.Thread(target=worker1)
    t2 = threading.Thread(target=worker2)
    # スレッドスタート
    t1.start()
    t2.start()
    print('started')
