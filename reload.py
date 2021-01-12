


path_src = 'source\\'
files = os.listdir(path_src)


for fl in files[0:len(files)-2]:
    k = fl.find('.')
    imp.reload(eval(fl[0:k]))
