import sys
import os

dirs=os.path.join("build","release")


def join(commands):
    if len(commands) == 1:
        return commands[0]
    if len(commands) == 0:
        return ''
    theCommand=''    
    for i in range(0,len(commands)-1):
        theCommand+=commands[i]+' && '
    theCommand+=commands[len(commands)-1]
    print(theCommand)
    return theCommand

for r, d, f in os.walk("build", topdown=False):
    for n in f:
        os.remove(os.path.join(r, n))
    for n in d:
        os.rmdir(os.path.join(r, n))

os.system(join(['mkdir '+dirs,'cd '+dirs,'cmake ..\.. -GNinja','ninja']))