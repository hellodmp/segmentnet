import time
import os
from multiprocessing import Process, Queue

def prepareDataThread(dataQueue, proc_id):
    for i in range(10):
        dataQueue.put(tuple((os.getpid(),i)))
        #time.sleep(10)

def processThread(dataQueue):
    for i in range(100):
        [proc_id, index] = dataQueue.get()
        print proc_id, index
        if (i+1)%10 == 0:
            print ""
        time.sleep(1)


if __name__ == "__main__":
    dataQueue = Queue(30)  # max 50 images in queue
    dataPreparation = [None] * 10
    # thread creation
    for proc in range(0, 10):
        dataPreparation[proc] = Process(target=prepareDataThread, args=(dataQueue,proc))
        dataPreparation[proc].daemon = True
        dataPreparation[proc].start()
    processThread(dataQueue)