import csv
import json


class Parser:

    def csv_to_json(self, local_file_path):
        data = self.read_csv(local_file_path)
        return json.dumps(data)

    def read_csv(self, local_file_path):
        csv_rows = []
        with open(file=local_file_path) as csv_file:
            reader = csv.DictReader(csv_file)
            title = reader.fieldnames
            for row in reader:
                csv_rows.extend([{title[i]: row[title[i]] for i in range(len(title))}])
            return csv_rows

    def write_csv(self, file_data):
        pass


# parser = Parser()
# print(parser.csv_to_json(local_file_path='/Users/keenan/Downloads/online_orders_20180105063310.csv'))
