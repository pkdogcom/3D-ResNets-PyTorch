import csv
import glob


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def load_n_frames(n_frames_path):
    n_frames_map = {}
    with open(n_frames_path, 'r') as f:
        for line in f.readlines():
            video_name, n_frames = line.rsplit(' ', 1)
            n_frames_map[video_name] = int(n_frames)
    print('Loaded {} n_frames'.format(len(n_frames_map)))
    return n_frames_map


def search_dir(root_path, depth=2):
    if not root_path.endswith('/'):
        root_path = root_path + '/'
    search_pattern = (root_path + '*/' * depth)[:-1] # Remove the last '/'
    return [path[len(root_path):] for path in glob.glob(search_pattern)]


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().data[0]

    return n_correct_elems / batch_size
