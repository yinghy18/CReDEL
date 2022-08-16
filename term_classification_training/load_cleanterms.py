import pandas as pd
from tqdm import tqdm


class CLEANTERMS(object):
    def __init__(self, path):
        self.df = pd.read_csv(path, '\t')
        self.cui2id = {}
        self.str2id = {}

        print(f"Load cleanterms from **{path}**")
        cui_list = self.df['cui'].tolist()
        str_lower_list = self.df['str.lower'].tolist()
        self.cui2id = {cui:idx for idx, cui in enumerate(cui_list)}
        self.str2id = {}
        for idx, str_lower in enumerate(str_lower_list):
            if not str_lower in self.str2id:
                self.str2id[str_lower] = []
            self.str2id[str_lower].append(idx)

    def str2cui(self, string):
        string = string.strip().lower()
        id_list = self.str2id[string]
        cuis = set([self.df['cui'][id] for id in id_list])
        return cuis

    def str2sty(self, string):
        cuis = self.str2cui(string)
        ids = set()
        for cui in cuis:
            ids.update([self.cui2id[cui]])
        stys = set()
        for id in ids:
            stys.update(self.df['sty'][id].split('|'))
        return list(stys)

    def str2sty_n(self, string):
        return len(self.str2sty(string))

    def str2sgr_n(self, string):
        cuis = self.str2cui(string)
        ids = set()
        for cui in cuis:
            ids.update([self.cui2id[cui]])
        sgr_n = []
        for id in ids:
            sgr_n.append(self.df['sgr.n'][id])
        return max(sgr_n)

    def str2sgr(self, string):
        cuis = self.str2cui(string)
        ids = set()
        for cui in cuis:
            ids.update([self.cui2id[cui]])
        sgrs = set()
        for id in ids:
            sgrs.update([self.df['sgr'][id]])
        return list(sgrs)

    def is_short_upper(self, string):
        string = string.strip().lower()
        id_list = self.str2id[string]
        for id in id_list:
            if self.df['short.upper'][id] == 1:
                return True
        return False
        
if __name__ == "__main__":
    cleanterms = CLEANTERMS('./dict/cleanterms4.txt')
    print(cleanterms.str2cui('cough'))
    print(cleanterms.str2sty('cough'))
    print(cleanterms.str2sgr_n('cough'))
    print(cleanterms.str2sgr('cough'))
    print(cleanterms.is_short_upper('cough'))
    import ipdb; ipdb.set_trace()
