import os
# os.system('cmd /k "cd C:/Users/youss/OneDrive/Desktop/GP/CyKit-master/CyKit-master/Py3"')


import threading

class mythread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        p1 = os
        p1.system('cmd /c "cd C:/Users/youss/OneDrive/Desktop/GP/CyKit-master/CyKit-master/Py3 && C:/Users/youss/AppData/Local/Programs/Python/Python37/python.exe ./CyKIT.py 127.0.0.1 54123 6 generic+noheader"')
        # p1.system('cmd /k "C:/Users/youss/AppData/Local/Programs/Python/Python37/python.exe ./CyKIT.py 127.0.0.1 54123 6 generic+noheader"')




thread1 = mythread()
thread1.start()

p2=os
p2.system('cmd /c "cd C:/Users/youss/OneDrive/Desktop/GP/CyKit-master && python tcpmy.py"')
# p2.system('cmd /k "python tcpmy.py"')
