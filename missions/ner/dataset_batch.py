# -*- coding: utf-8 -*-

import numpy as np
import os
import re
import operator
import pickle
import sys

class Dataset:
    def __init__(self, parameter, extern_data, Use_pretrained = False):
        self.parameter = parameter
        self.extern_data = extern_data

        if parameter["mode"] == "train" and not os.path.exists(parameter["necessary_file"]):
            self._make_necessary_data_by_train_data()
        else:
            with open(parameter["necessary_file"], 'rb') as f:
                self.necessary_data = pickle.load(f)
        if Use_pretrained == True:
            print('2')
            cnt = 0
            data = []
            error_voc = 0
            with open('./punct_dim200_3.vec', 'r') as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    data.append(line.split('\n')[0])

            voca_size, emb_dim = data[0].split()
            voca_size, emb_dim = int(voca_size), int(emb_dim)
            word = []
            embedding = []
            word.append('<UNK>')
            embedding.append(np.random.uniform(-0.25, 0.25, emb_dim))
            print('3')
            for i in range(1, len(data)):
                temp_embedding = []
                temp = data[i].split()

                if len(temp) != emb_dim+1:
                    error_voc += 1
                    continue

                for i in range(0, emb_dim):
                    temp_embedding.append(float(temp[i + 1]))
                word.append(temp[0])
                embedding.append(temp_embedding)
            pretrained_embedding = [0] * len(self.necessary_data["word"])
            print('4')
            """for i in range(1, len(data)):
                temp_embedding = []
                temp = data[i].split()
                if len(temp) != 200:
                    error_voc += 1
                    continue
                for i in range(0, emb_dim):
                    temp_embedding.append(float(temp[i + 1]))
                if temp[0] in self.necessary_data["word"]:
                    cnt+=1
                    word.append(temp[0])
                    embedding[(self.necessary_data["word"])[temp[0]]] = temp_embedding """
            for k, v in self.necessary_data["word"].items():
                if k in word:
                    cnt += 1
                    pretrained_embedding[v] = embedding[word.index(k)]
                else:
                    pretrained_embedding[v] = embedding[0]
            with open('pre_fasttext_200.pkl', 'wb') as f:
                pickle.dump(pretrained_embedding, f)
            print('The num of word in pretraiedn_WordEmbedding',cnt, 'And total voca size is ', len(self.necessary_data["word"]))

            self.pretrained_fasttext = pretrained_embedding
            #return voca_size - error_voc, emb_dim, word, embedding
        self.parameter["embedding"] = [
                [ "word", len(self.necessary_data["word"]), parameter["word_embedding_size"]],
                [ "character", len(self.necessary_data["character"]), parameter["char_embedding_size"]],
                ["word_pretrain", len(self.necessary_data["word"]), parameter["pretrained_word_embedding_size"]]
            ]
        #WATCH OUT EMBEDDING at model.py -> build_model
        #necessary_data["word"] 어절 사전
        #,
        #        self.parameter["embedding"].append(["pretrained_word", len(self.necessary_data["word"]), parameter["pretrained_word_embedding_size"]])
        self.parameter["n_class"] = len(self.necessary_data["ner_tag"])

    def _make_necessary_data_by_train_data(self):
        necessary_data = {"word": {}, "character": {},
                          "ner_tag": {}, "ner_morph_tag": {}}

        for morphs, tags, ner_tag, ner_mor_list, ner_tag_list in self._read_data_file(extern_data=self.extern_data):
            for mor, tag in zip(morphs, tags):
                self._check_dictionary(necessary_data["word"], mor)

                for char in mor:
                    self._check_dictionary(necessary_data["character"], char)

            if type(ner_tag) is list:
                for ne in ner_tag:
                    if ne == "-": continue
                    self._check_dictionary(necessary_data["ner_tag"], ne + "_B")
                    self._check_dictionary(necessary_data["ner_tag"], ne + "_I")
            else:
                self._check_dictionary(necessary_data["ner_tag"], ner_tag + "_B")
                self._check_dictionary(necessary_data["ner_tag"], ner_tag + "_I")

            for nerMor, nerTag in zip(ner_mor_list, ner_tag_list):
                if nerTag == "-" or nerTag == "-_B": continue
                nerTag = nerTag.split("_")[0]
                self._check_dictionary(necessary_data["ner_morph_tag"], nerMor, nerTag)

        # 존재하는 어절 사전
        necessary_data["word"] = self._necessary_data_sorting_and_reverse_dict(necessary_data["word"], start=2)

        # 존재하는 음절 사전
        necessary_data["character"] = self._necessary_data_sorting_and_reverse_dict(necessary_data["character"], start=2)

        # 존재하는 NER 품사 태그 사전
        necessary_data["ner_tag"] = self._necessary_data_sorting_and_reverse_dict(necessary_data["ner_tag"], start=2, unk=False)
        self.ner_tag_size = len(necessary_data["ner_tag"])
        self.necessary_data = necessary_data

        # 존재하는 형태소 별 NER 품사 태그 비율 사전
        necessary_data["ner_morph_tag"] = self._necessary_data_sorting_and_reverse_dict(necessary_data["ner_morph_tag"], start=0, ner=True)

        with open(self.parameter["necessary_file"], 'wb') as f:
            pickle.dump(necessary_data, f)

    def make_input_data(self, extern_data=None):
        morphs = []
        ne_dicts = []
        characters = []
        labels = []
        sequence_lengths = []
        character_lengths = []

        if extern_data is not None:
            self.extern_data = extern_data

        temp = [[], [], []]
        # TAG 정보가 없는 경우에는 tag 자리에 mor 정보가 들어온다
        #ner_mor = ['비토리오','양일',...,'가율']
        #ner_tag = ['PER_B','DAT_B','-',...,'-']
        #tag = mor = ner_mor
        for mor, tag, _, ner_mor, ner_tag in self._read_data_file(pre=False, extern_data=self.extern_data):
            if tag != False:
                temp[0] += mor
                temp[1] += tag
                if len(ner_tag) == 0:
                    temp[2] += ['O'] * len(mor)
                elif len(ner_tag) == len(mor):
                    temp[2] = ner_tag
                else:
                    for i, m in enumerate(mor):
                        if m == ner_mor[0]:
                            break
                    ner_tag = ['O'] * i + ner_tag
                    ner_tag = ner_tag + ['O'] * (len(mor) - len(ner_tag))
                    temp[2] += ner_tag
            else:
                morph = [0] * self.parameter["sentence_length"]
                ne_dict = [[0.] * int(self.parameter["n_class"] / 2)] * self.parameter["sentence_length"]
                character = [[0] * self.parameter["word_length"]] * self.parameter["sentence_length"]
                character_length = [0] * self.parameter["sentence_length"]
                label = [0] * self.parameter["sentence_length"]

                if len(temp[0]) > self.parameter["sentence_length"]:
                    #sentence_length보다 크면 그냥 무시?
                    temp = [[], [], []]
                    continue

                sequence_lengths.append(len(temp[0]))
                for mor, tag, neTag, index in zip(temp[0], temp[1], temp[2], range(0, len(temp[0]))):
                    morph[index] = self._search_index_by_dict(self.necessary_data["word"], mor)
                    ne_dict[index] = self._search_index_by_dict(self.necessary_data["ner_morph_tag"], mor)
                    if neTag != "-" and neTag != "-_B":
                        label[index] = self._search_index_by_dict(self.necessary_data["ner_tag"], neTag)
                    sub_char = [0] * self.parameter["word_length"]
                    for i, char in enumerate(mor):
                        if i == self.parameter["word_length"]: 
                            i-=1
                            break
                        sub_char[i] = self._search_index_by_dict(self.necessary_data["character"], char)
                    character_length[index] = i+1
                    character[index] = sub_char

                morphs.append(morph)
                ne_dicts.append(ne_dict)
                characters.append(character)
                character_lengths.append(character_length)
                labels.append(label)

                temp = [[], [], []]

        self.morphs = np.array(morphs)
        self.ne_dicts = np.array(ne_dicts)
        self.characters = np.array(characters)
        self.sequence_lengths = np.array(sequence_lengths) #각 문장 길이를 저장
        self.character_lengths = np.array(character_lengths)
        self.labels = np.array(labels)

    def get_data_batch_size(self, n, train=True):
        if train:
            for i, step in enumerate(range(0, self.parameter["train_lines"], n)):
                if len(self.morphs[step:step + n]) == n:
                    yield self.morphs[step:step+n], self.ne_dicts[step:step+n], self.characters[step:step+n], \
                        self.sequence_lengths[step:step+n], self.character_lengths[step:step+n], \
                        self.labels[step:step+n], i
        else:
            for i, step in enumerate(range(0, self.parameter["train_lines"], n)):
                if len(self.morphs[step:step+n]) == n:
                    yield self.morphs[step:step+n], self.ne_dicts[step:step+n], self.characters[step:step+n], \
                        self.sequence_lengths[step:step+n], self.character_lengths[step:step+n], \
                        self.labels[step:step+n], i

    def _search_index_by_dict(self, dict, key):
        if key in dict:
            return dict[key]
        else:
            if "UNK" in dict:
                return dict["UNK"]
            else:
                temp = [0.0] * int(self.parameter["n_class"] / 2)
                temp[0] = 1.0
                return temp

    def _read_data_file(self, pre=True, extern_data=None):
        if extern_data is not None:
            return self._read_extern_data_file(pre, self.extern_data)

    def _read_extern_data_file(self, pre=True, extern_data=None):
        cntLine = 0
        for sentence in extern_data:
            morphs = []
            tags = []
            ner_tag = []
            ner_mor_list = []
            for morph in sentence[1]:
                morphs.append(morph)
                tags.append(morph)
                ner_mor_list.append(morph)
            seq_len = len(morphs)

            ner_tag_list = ['O'] * seq_len
            for index, ne in enumerate(sentence[2]):
                ner_tag.append(ne.split("_")[0])
                ner_tag_list[index] = ne

            yield morphs, tags, ner_tag, ner_mor_list, ner_tag_list
            cntLine += 1
            if pre == False:
                yield [], False, False, False, False
            if cntLine % 1000 == 0:
                sys.stderr.write("%d Lines .... \r" % ( cntLine ))

                if self.parameter["train_lines"] < cntLine:
                    break

    def _check_dictionary(self, dict, data, value=0):
        if type(value) is int:
            if not data in dict:
                dict[data] = value
        elif type(value) is str:
            if not value in dict:
                dict[data] = {value: 1}
            else:
                if value in dict[data]:
                    dict[data][value] += 1
                else:
                    dict[data][value] = 1

    def _necessary_data_sorting_and_reverse_dict(self, dict, start=1, unk=True, ner=False):
        dict_temp = {}
        index = start

        if start == 2:
            dict_temp["PAD"] = 0
            if unk:
                dict_temp["UNK"] = 1
            else:
                dict_temp["O"] = 1
        elif start == 1:
            dict_temp["PAD"] = 0

        for key in sorted(dict.items(), key=operator.itemgetter(0), reverse=False):
            if ner:
                items = np.zeros(int(self.ner_tag_size / 2))
                for i in key[1]:
                    items[int(self.necessary_data["ner_tag"][i + "_B"] / 2)] = dict[key[0]][i]
                dict_temp[key[0]] = items / np.sum(items)
            else:
                dict_temp[key[0]] = index
                index += 1

        return dict_temp


if __name__ == "__main__":
    dataset = Dataset({"input_dir": "data/NER.sample.txt"})
