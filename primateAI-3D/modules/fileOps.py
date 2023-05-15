import os
import errno

__author__ = 'thamp'

def mkdir_p(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: raise


def touchFile(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)

def windowsPathToLinuxPath(windowsPath):
    linuxPath = windowsPath.replace("\\", "/")
    linuxPath = linuxPath.replace("Z:", "/home/thamp")
    return linuxPath


def linuxPathToWindowsPath(linuxPath):
    linuxPath = linuxPath.replace("/home/thamp", "Z:")
    linuxPath = linuxPath.replace("/", "\\")
    return linuxPath


def localPathToClusterPath(localPath):
    return localPath.replace("/mnt", "")
    

def clusterPathToLocalPath(clusterPath):
    return "/mnt" + clusterPath


def catTxtFiles(filePaths, outputFilePath=None, dropHeaderLine=False):
    lines = []
    for i, filePath in enumerate(filePaths):
        fileObj = open(filePath)
        for j, line in enumerate(fileObj):
            if dropHeaderLine and i>0 and j==0:
                continue
            lines.append(line.rstrip("\n"))
        fileObj.close()

    if outputFilePath != None:
        outFile = open(outputFilePath, 'w')
        outFile.write("\n".join(lines)+"\n")
        outFile.close()
    return lines

def prependLine(line, filePath):
    
    with open(filePath) as fileObj:    
        fileContent = fileObj.read()
    
    newFile = open(filePath, 'w')
    newFile.write(line)
    newFile.write(fileContent)
    newFile.close()
    
    
