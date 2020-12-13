import yaml
import requests
import gzip
from collections import namedtuple


class MNIST_Loader:
    def __init__(self, confPath='config.yml'):
        self.ImagesFileHeader = namedtuple('ImagesFileHeader', ['magic', 'images_num', 'rows_num', 'columns_num'])
        self.LabelsFileHeader = namedtuple('LabelsFileHeader', ['magic', 'labels_num'])
        self.Files2download = namedtuple('Files2download', ['file_name', 'description', 'url'])
        self.confPath = confPath

        with open(confPath) as f:
            config = yaml.safe_load(f)
            self.data2download_MNIST = tuple(
                self.Files2download(*entry) for entry in config['DEFAULT']['data2download_MNIST'])

    def check_consistensy(self):
        pass

    def download(self):
        data = []
        for i, entry in enumerate(self.data2download_MNIST):
            r = requests.get(entry.url, stream=True)
            if r.status_code == requests.codes.ok:
                if not i:
                    print('Statistics on downloaded files:')
                print('\t{fileName}: {description} ({size} bytes)'.format(
                    fileName=entry.file_name,
                    description=entry.description,
                    size=r.headers['content-length']))
                decompressed = gzip.decompress(r.raw.read())
                data.append(decompressed)
        return data

    def convert_data_to_ndarrays(self):
        pass


if __name__ == '__main__':
    loader = MNIST_Loader()
    data = loader.download()
    # print(data)
