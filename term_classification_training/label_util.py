import json

def get_entity_type_from_train_file():
    sty_set = set()
    sty2sgr = {}
    with open("./dict/cleanterms5.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
    for line in lines:
        stys = line.split("\t")[2]
        sty_set.update(stys.split("|"))
        sgr = line.split("\t")[3]
        for sty in stys.split("|"):
            sty2sgr[sty] = sgr
    sty_list = list(sty_set)
    sty_list.sort()
    sty2id = {sty:idx for idx, sty in enumerate(sty_list)}
    with open("./dict/entity_type.json", "w", encoding="utf-8") as f:
        json.dump(sty2id, f)
    with open("./dict/entity_group.json", "w", encoding="utf-8") as f:
        json.dump(sty2sgr, f)


def get_entity_type_from_json():
    with open("./dict/entity_type.json", "r", encoding="utf-8") as f:
        sty2id = json.loads(f.readline())
    with open("./dict/entity_group.json", "r", encoding="utf-8") as f:
        sty2sgr = json.loads(f.readline())
    return sty2id, sty2sgr


def check_entity_type():
    sty_set = set()
    sty2sgr = {}
    with open("./dict/cleanterms5.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
    for line in lines:
        stys = line.split("\t")[2]
        sty_set.update(stys.split("|"))
        sgr = line.split("\t")[3]
        for sty in stys.split("|"):
            if not sty in sty2sgr:
                sty2sgr[sty] = set()
            sty2sgr[sty].update([sgr])
    print(sty2sgr)

if __name__ == "__main__":
    check_entity_type()

