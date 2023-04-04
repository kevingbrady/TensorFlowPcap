from src.CsvReader import CsvReader


class ClassAnalyzer:

    def __init__(self, file):

        self.dataset_size = 0
        self.benign = {
            'count': 0,
            'percent': 0.0
        }
        self.malicious = {
            'count': 0,
            'percent': 0.0
        }

        self.get_class_values(file)

    def get_class_values(self, file):

        reader = CsvReader(file)
        data_generator = reader.read_file()

        for chunk in data_generator:
            counts = chunk['Target'].value_counts()

            if 1 in counts:
                self.malicious['count'] += counts[1]
            if 0 in counts:
                self.benign['count'] += counts[0]

            self.dataset_size += chunk['Target'].size

        self.malicious['percent'] = self.malicious['count'] / self.dataset_size
        self.benign['percent'] = self.benign['count'] / self.dataset_size

    def print_class_analysis_output(self):
        print('CLass Values:  [Malicious] %d  [Benign] %d' % (self.malicious['count'], self.benign['count']))
        print('Class Balance: [Malicious] %.3f [Benign] %.3f' % (self.malicious['percent'], self.benign['percent']))
        print('Dataset Size: %d' % self.dataset_size)


