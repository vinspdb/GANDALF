from jinja2 import Template
from utility.skeleton import skeleton
import pickle

class SentinelToText():
    def __init__(self, df, bl, df_type):
        self.__bl = bl
        self.__df = df
        self.__df_type = df_type
        self.__list_history = []
        self.__gen_descritions()

    def __gen_descritions(self):
        print(skeleton[self.__bl]['template'])
        history_template = Template(skeleton[self.__bl]['template'])
        for index, row in self.__df.iterrows():
            dict_hist = {}
            for v in skeleton[self.__bl]['feature']:
                dict_hist[v] = int(row[v])

            history_text = history_template.render(dict_hist)
            self.__list_history.append(history_text)
        self.__serialize()

    def __serialize(self):
        with open('CZ_200_Settembre/'+self.__df_type+'/'+self.__df_type+'.pkl', 'wb') as f:
            pickle.dump(self.__list_history, f)
