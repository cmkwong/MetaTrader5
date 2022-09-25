import os

from mt5f.BaseMt5 import BaseMt5


class CsvWriterHelper(BaseMt5):
    def __init__(self, csv_save_path='', csv_file_names=None, append_checkpoint=None):
        super(CsvWriterHelper, self).__init__()
        # for output csv file
        self.csv_save_path = csv_save_path
        self.append_checkpoint = append_checkpoint
        self._register_csv_file_datas(csv_file_names) # store the text, if no need to store for specific loader file, return empty dictionary
        self._register_csv_txt_append_count(csv_file_names) # store the appending count for specific loader file, if no need to append, return empty dictionary
        self._appended_text = False # if this class have been appended text, then set as True

    def __exit__(self, *args):
        if self._appended_text: # if there is a appended text before, evacuate the rest of loader before exit
            self.evacuate_csv_file_datas()
        self.disconnect_server()

    def _register_csv_file_datas(self, csv_file_names):
        self._csv_file_datas = {}
        if csv_file_names is not None:
            for csv_file_name in csv_file_names:
                self._csv_file_datas[csv_file_name] = ''

    def _register_csv_txt_append_count(self, csv_file_names):
        self._csv_txt_append_count = {}
        if csv_file_names is not None:
            for csv_file_name in csv_file_names:
                self._csv_txt_append_count[csv_file_name] = 0

    def append_txt(self, csv_text, csv_file_name):
        """
        :param csv_text: text being stored and write later
        :param csv_file_name: file name that being output or append
        """
        # check if csv file name is registered
        if csv_file_name not in list(self._csv_file_datas.keys()):
            raise Exception("{} is not registered.".format(csv_file_name))

        # if first time in appending text, do NOT discard the header
        if self._csv_txt_append_count[csv_file_name] > 0:
            csv_text = csv_text.split('\n', 1)[1] # discard first row that is header
        # append the text
        self._csv_file_datas[csv_file_name] += csv_text

        # if reach the checkpoint, write and empty the loader as csv
        if self._csv_txt_append_count[csv_file_name] % self.append_checkpoint == 0:
            self.evacuate_csv_file_datas()
        self._csv_txt_append_count[csv_file_name] += 1
        self._appended_text = True

    def evacuate_csv_file_datas(self):
        for csv_file_name, csv_text in self._csv_file_datas.items():
            if len(csv_text) > 0:
                print("Writing {} ... ".format(csv_file_name), end='')
                with open(os.path.join(self.csv_save_path, csv_file_name), 'a+') as f:
                    f.write(csv_text)
                    f.close()
                    print('OK')
                # empty the datas
            self._csv_file_datas[csv_file_name] = ''