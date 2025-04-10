import glob
import json
import os
import shutil

LABEL = {
    'neu': '01',  #: 'neutral',
    # 'fru': '02',  #: 'calm',
    # 'hap': '03',  #: 'happy',
    'sad': '04',  #: 'sad',
    'ang': '05',  #: 'angry',
    # 'fea': '06',  #: 'fearful',
    'exc': '07',  #: 'excitement','disgust',
    # 'sur': '08',  #: 'surprised'
    # 'xxx': '09',  #: 'other'
}

dict_ses = {"Ses01F": "Session1", "Ses01M": "Session1", "Ses02F": "Session2",
            "Ses02M": "Session2", "Ses03F": "Session3", "Ses03M": "Session3",
            "Ses04F": "Session4", "Ses04M": "Session4", "Ses05F": "Session5",
            "Ses05M": "Session5", }

PATH_TXT = glob.glob(r"E:/pytorch_project/speech_emotion_recognition/IEMOCAP_full_release/*/dialog/EmoEvaluation/S*.txt")
PATH_WAV = glob.glob(r"E:/pytorch_project/speech_emotion_recognition/IEMOCAP_full_release/*/sentences/wav/*/S*.wav")
# PATH_TXT = glob.glob("F:/IEMOCAPDATA/IEMOCAP_full_release/*/dialog/EmoEvaluation/S*.txt")
# PATH_WAV = glob.glob("F:/IEMOCAPDATA/IEMOCAP_full_release/*/sentences/wav/*/S*.wav")

PAIR = {}


def getPair():
    for path in PATH_TXT:
        with open(path, 'r') as f:
            fr = f.read().split("\t")
            for i in range(len(fr)):
                if (fr[i] in LABEL):
                    PAIR[fr[i - 1]] = fr[i]


def rename():
    for i in PATH_WAV:
        for j in PAIR:
            if (os.path.basename(i)[:-4] == j):
                k = j.split('_')
                if (len(k) == 3):
                    name = 'E:/pytorch_project/speech_emotion_recognition/interspeech21_emotion/path_to_wavs' + '/' + dict_ses[k[0]] + '/' + k[0] + '-' + k[1] + '-' + LABEL[PAIR[j]] + '-01-' + k[2] + '.wav'
                    # name = os.path.dirname(i) + '/' + k[0] + '-' + k[1] + '-' + LABEL[PAIR[j]] + '-01-' + k[2] + '.wav'
                    # os.rename(src=i, dst=name)
                    # shutil.copyfile(i, name)
                    print(name)
                    '''
                    Ses01F_impro01_F000.wav
                    k[0]:Ses01F
                    k[1]:impro01
                    k[2]:F000
                    Ses01F-impro01-XX-01-F000.wav
                    '''
                elif (len(k) == 4):
                    # name = 'F:/IEMOCAP_full_release/data' + '/' + k[0] + '-' + k[1] + '-' + LABEL[PAIR[j]] + '-01-' + k[2] + '_' + \
                    #        k[
                    #            3] + '.wav'
                    name = os.path.dirname(i) + '/' + dict_ses[k[0]] + k[0] + '-' + k[1] + '-' + LABEL[PAIR[j]] + '-01-' + k[2] + '_' + \
                           k[3] + '.wav'
                    # os.rename(src=i, dst=name)
                    shutil.copyfile(i, name)
                    # print(name)
                    '''
                    Ses03M_script03_2_F032.wav
                    k[0]:Ses03M
                    k[1]:script03
                    k[2]:2
                    k[3]:F032
                    Ses03M-script03-XX-01-2_F032.wav
                    '''


if __name__ == '__main__':
    # pairPath = "F:/IEMOCAP_full_release/pair.json"
    pairPath = r"./pair.json"
    # if (os.path.exists(pairPath)):
    #     with open(pairPath, 'r') as f:
    #         PAIR = json.load(f)
    # else:
    #     getPair()
    #     with open(pairPath, 'w') as f:
    #         json.dump(obj=PAIR, fp=f)
    # rename()
    # print('done!')

    # pairPath = r"./pair.json"
    # if (os.path.exists(pairPath)):
    #     with open(pairPath, 'r') as f:
    #         PAIR = json.load(f)
    # else:
    getPair()
    with open(pairPath, 'w') as f:
        json.dump(obj=PAIR, fp=f)
    rename()
    print('done!')