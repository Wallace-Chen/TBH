import os,sys

folder = r"E:\Users\yuan\MasterThesis\TBH\data\test\out_NUS-WIDE_incv3-keras"

f_list = [os.path.join(folder, "pics", f) for f in os.listdir(os.path.join(folder, "pics")) if f.endswith('jpg') ]

tot = 0
deleted = 0
for f in f_list:
    tot += 1
    if tot%1000 == 0: print("{} files processed".format(tot))
    size = os.path.getsize(f)
    if size<1000:
        deleted += 1
        os.remove(f)

print("There are {} totally files, and {} files have been deleted.".format(tot, deleted))
