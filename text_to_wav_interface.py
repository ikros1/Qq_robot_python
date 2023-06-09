# -*- coding: utf-8 -*-
import argparse
import math
import os
import random
import wave
from PIL import Image, ImageTk
import tkinter as tk
import time
import numpy as np
import torch
from dotenv import load_dotenv
from torch import no_grad, LongTensor
import commons
import utils
from models_infer import SynthesizerTrn
from text import text_to_sequence
from face_draw import EllipseApp




device = "cuda:0" if torch.cuda.is_available() else "cpu"
language_marks = {
    "Japanese": "",
    "日本語": "[JA]",
    "简体中文": "[ZH]",
    "English": "[EN]",
    "Mix": "",
}
lang = ['日本語', '简体中文', 'English', 'Mix']
mouse_turn = {' ': '#', 'a': '8', 'g': '8', 'i': '8', 'ɯ': '8', 'e': '8', 'o': '8', 'k': '8', 'z': '8', 'ʃ': '8',
              'd': '8', 'ʑ': '8', 's': '8', 't': '8', 'n': '8', 'p': '6', 'ɸ': '10', 'm': '6', 'j': '8', 'w': '8',
              'b': '6', 'h': '9', 'ɾ': '9', 'ŋ': '#', '↑': '#', '^': '#', '!': '#', '': '#', '#': '#', 'N': '#',
              '↓': '#', 'ç': "#", '?': '_', ':': '_', '。': '_', '、': '*', '？': '#', '*': '#', '_': '#', '.': '_',
              ',': '_'}


def display_images(face_id):
    app = EllipseApp(face_arr=face_id)


def match_dict(s, d):
    res = []
    for c in s:
        if c in d:
            if "#" == d[c]:
                res.append(res[-1])
            else:
                adr = d[c] + ".png"
                res.append(adr)
        else:
            res.append(f'匹配失败{c}')
    print(res)
    print(len(res))
    return res


def get_text(text, hps, is_symbol):
    text_norm, clean_text_s, clean_text_s_len = text_to_sequence(text, hps.symbols,
                                                                 [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm, clean_text_s, clean_text_s_len


def create_tts_fn(model, hps, speaker_ids):
    def tts_fn(text, speaker, language, speed, voice_list):
        if language is not None:
            text = language_marks[language] + text + language_marks[language]
        # speaker_id = speaker_ids[speaker]
        for i in voice_list.keys():
            if speaker in i:
                speaker_id = voice_list[i]
        stn_tst, clean_text_s, clean_text_s_len = get_text(text, hps, False)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return audio

    return tts_fn


class Core_tts_ika:
    def __init__(self):
        self.voice_list = []
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_dir", default="./G_latest.pth", help="directory to your fine-tuned model")
        parser.add_argument("--config_dir", default="./finetune_speaker.json",
                            help="directory to your model config file")
        parser.add_argument("--share", default=False, help="make link public (used in colab)")

        args = parser.parse_args()
        # print(args)
        hps = utils.get_hparams_from_file(args.config_dir)

        net_g = SynthesizerTrn(
            len(hps.symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model).to(device)
        _ = net_g.eval()

        _ = utils.load_checkpoint(args.model_dir, net_g, None)
        speaker_ids = hps.speakers
        self.voice_list = speaker_ids
        self.tts_fn = create_tts_fn(net_g, hps, speaker_ids)

    def tts_vo(self, text, speaker, language, speed,file_path):
        audio_data = self.tts_fn(text, speaker, language, speed, self.voice_list)
        face_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 170)
        face_data = face_data.astype(np.float64)
        face_data = np.abs(face_data)
        face_data = face_data[::441]
        n = 8
        pad_data = np.pad(face_data, (n // 2, n // 2), mode='edge')
        smooth_data = np.convolve(pad_data, np.ones(n) / n, mode='valid')
        smooth_data = smooth_data.astype('int16')
        audio_file = wave.open(file_path, 'w')
        audio_file.setparams((1, 2, 22050, 0, 'NONE', 'NONE'))
        audio_data = np.asarray(audio_data, dtype=np.float32)
        audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
        #print(audio_data)
        audio_file.writeframes(audio_data.tobytes())
        audio_file.close()
        return smooth_data


if __name__ == "__main__":
    t_k = Core_tts_ika()
    text = "太陽は、私たちが住む地球に最も近い星のひとつであり、太陽系の中心に位置しています。太陽は、直径が約1,391,000キロメートルで、地球の約109倍の大きさを持ち、質量は地球の約333,000倍です。太陽は、水素とヘリウムなどの軽い元素からなるプラズマ状態の物質から構成されており、その中心部の温度は約1,500万度にも達すると言われています。この高温により、太陽は常に光と熱を放射しています。"
    speaker = "ikaros"
    language = "日本語"
    # language = "简体中文"
    speed = 0.8
    front_path = "go-cqhttp/data/voices/"
    # 通过时间戳赋予随机名字
    file_name = str(int(time.time()))+str(random.randint(0, 1000))
    file_back = ".wav"
    file_name_all = front_path + file_name + file_back
    t_k.tts_vo(text=text, speaker=speaker, language=language, speed=speed, file_path=file_name_all)

