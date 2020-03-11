
import glob

dir= r"C:\Users\gad\Downloads\Compressed\lanenet-lane-detection\data\data_records"
paths= glob.glob('{:s}/{:s}.tfrecords'.format(dir, 'train'))

print(paths)