#!/usr/bin/env python3
# Driver for whisper.sif
import shutil
import subprocess
import sys
import tempfile
import argparse
import logging
from pathlib import Path


# we need the amp packages for some things, but alas galaxy has overwritten the
# carefully crafted PYTHONPATH that included them.  So let's get them back if
# we can.
import os
if 'AMP_ROOT' in os.environ:
    sys.path.append(os.environ['AMP_ROOT'] + "/amp_bootstrap")

import amp.logging
import amp.gpu
from amp.timeutils import hhmmss2timestamp, timestamp2hhmmss
from amp.fileutils import read_json_file, write_json_file
from amp.vtt_helper import gen_vtt, words2phrases


whisper_languages = [
    'Auto', 'Afrikaans', 'Albanian', 'Amharic', 'Arabic', 'Armenian', 'Assamese', 'Azerbaijani',
    'Bashkir', 'Basque', 'Belarusian', 'Bengali', 'Bosnian', 'Breton', 'Bulgarian', 'Burmese',
    'Castilian', 'Catalan', 'Chinese', 'Croatian', 'Czech',
    'Danish', 'Dutch',
    'English', 'Estonian', 'Faroese', 'Finnish', 'Flemish', 'French',
    'Galician', 'Georgian', 'German', 'Greek', 'Gujarati',
    'Haitian', 'Haitian Creole', 'Hausa', 'Hawaiian', 'Hebrew', 'Hindi', 'Hungarian',
    'Icelandic', 'Indonesian', 'Italian',
    'Japanese', 'Javanese',
    'Kannada', 'Kazakh', 'Khmer', 'Korean',
    'Lao', 'Latin', 'Latvian', 'Letzeburgesch', 'Lingala', 'Lithuanian', 'Luxembourgish',
    'Macedonian', 'Malagasy', 'Malay', 'Malayalam', 'Maltese', 'Maori', 'Marathi', 'Moldavian',
    'Moldovan', 'Mongolian', 'Myanmar',
    'Nepali', 'Norwegian', 'Nynorsk',
    'Occitan',
    'Panjabi', 'Pashto', 'Persian', 'Polish', 'Portuguese', 'Punjabi', 'Pushto',
    'Romanian', 'Russian',
    'Sanskrit', 'Serbian', 'Shona', 'Sindhi', 'Sinhala', 'Sinhalese', 'Slovak', 'Slovenian',
    'Somali', 'Spanish', 'Sundanese', 'Swahili', 'Swedish',
    'Tagalog', 'Tajik', 'Tamil', 'Tatar', 'Telugu', 'Thai', 'Tibetan', 'Turkish', 'Turkmen',
    'Ukrainian', 'Urdu', 'Uzbek',
    'Valencian', 'Vietnamese',
    'Welsh',
    'Yiddish', 'Yoruba'
]
# I originally set this when I was loading the models into the container but
# in the interest of size, they're now loaded to the user's home directory.
whisper_model_dir = "/opt/whisper"
whisper_models = ['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en',
                  'medium', 'medium.en', 'large-v1', 'large-v2', 'large']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true", help="Turn on debugging")
    parser.add_argument("input_media", help="Input media file")
    parser.add_argument("--transcript_json", type=str, help="Output Whisper Transcript JSON file")
    parser.add_argument("--transcript_text", type=str, help="Output Whisper Transcript Text file")
    parser.add_argument("--amp_transcript", type=str, help="Output AMP Transcript File")
    parser.add_argument("--amp_diarization", type=str, help="Output AMP Diarization FIle")
    parser.add_argument("--web_vtt", type=str, help="WebVTT output")    
    parser.add_argument("--language", choices=whisper_languages, default="Auto", help="Audio Language")
    parser.add_argument("--model", choices=whisper_models, default='small', help="Language model to use")
    parser.add_argument("--cpuonly", default=False, action="store_true", help="Force CPU only computation")
    parser.add_argument("--vtt_phrase_gap", type=float, default=1.5, help="Minimum gap between VTT subtitle phrases")
    parser.add_argument("--vtt_max_duration", type=float, default=3.0, help="Maximum duration of VTT subtitle")
    args = parser.parse_args()    
    amp.logging.setup_logging("aws_transcribe", args.debug)    
    logging.info(f"Starting with args {args}")

    if args.transcript_json is None and args.transcript_text is None and args.amp_transcript is None and args.web_vtt is None and args.amp_diarization is None:
        logging.error("You must select an output of some sort!")
        exit(1)

    sif = Path(sys.path[0], "whisper.sif")
    if not sif.exists():
        logging.error(f"Whisper SIF file {sif!s} not found")
        exit(1)

    has_gpu = False
    if not args.cpuonly and amp.gpu.has_gpu('nvidia'):
        runcmd = ['apptainer', 'run', '--nv', str(sif)]
        has_gpu = True
    else:
        runcmd = [str(sif)]

    with tempfile.TemporaryDirectory(prefix="whisper-") as tmpdir:
        logging.debug(f"Temporary directory: {tmpdir}")
        whisper_args = ["--output_dir", tmpdir,
                        "--model", args.model,
                        "--word_timestamps", "True",
                        args.input_media]
        if args.language != "Auto":
            whisper_args.extend(['--language', args.language])
        
        cmd = [*runcmd, *whisper_args]
        logging.info(f"Whisper command: {cmd}")
        try:
            if has_gpu:
                with amp.gpu.ExclusiveGPU('nvidia') as g:
                    logging.info(f"Acquired device {g.name}")
                    subprocess.run(cmd, check=True)
            else:
                subprocess.run(cmd, check=True)
        except Exception as e:
            logging.exception(f"Failed to transcribe: {e}")
            exit(1)
                        
        try:
            # go through each of the requested outputs and gather them.
            if args.transcript_json:
                shutil.copy(get_file_by_ext(tmpdir, 'json'), args.transcript_json)

            if args.transcript_text:
                shutil.copy(get_file_by_ext(tmpdir, 'txt'), args.transcript_text)

            if args.web_vtt:
                generate_webvtt(get_file_by_ext(tmpdir, 'json'), args.web_vtt, 
                                args.vtt_phrase_gap, args.vtt_max_duration)

            if args.amp_transcript:
                generate_amp_transcript(get_file_by_ext(tmpdir, 'json'), args.amp_transcript, args.input_media)

            if args.amp_diarization:
                generate_amp_diarization(get_file_by_ext(tmpdir, 'json'), args.amp_diarization, args.input_media)

        except Exception as e:
            logging.exception(f"Failed to gather outputs: {e}")
            exit(1)

    logging.info("Finished!")


def get_file_by_ext(path, ext):
    "Find the first file with the given extension"
    files = list(Path(path).glob(f"*.{ext}"))
    if files:
        return files[0]
    else:
        raise FileNotFoundError(f"Cannot find file with extension {ext} in {path}")


def generate_webvtt(whisper_json, output_vtt, phrase_gap, max_duration):
    "Generate a VTT without underlines and with reasonable timestamps"
    # original version parsed the output VTT, but it makes more sense to
    # just use the json that was generated and generate it fresh.    
    data = read_json_file(whisper_json)
    # the output of whisper's cli puts all of the words into their segments
    # but we need a single word list for words2phrases
    words = []
    for s in data['segments']:
        words.extend(s['words'])

    # also, for some reason whisper likes to prepend spaces to the words so
    # let's strip that too
    for w in words:
        w['word'] = w['word'].strip()

    phrases = words2phrases(words, phrase_gap=phrase_gap, 
                            max_duration=max_duration)

    with open(output_vtt, "w") as o:
        o.write(gen_vtt(phrases))


def generate_amp_transcript(whisper_file, amp_file, input_media):
    # convert the native json transcript to an amp transcript
    data = read_json_file(whisper_file)
    amp_transcript = {
        'media': {
            'filename': input_media,
            'duration': 0
        },
        'results': {
            'transcript': data['text'].strip(),
            'words': []
        }
    }                
    duration = 0
    offset = 0
    for seg in data['segments']:
        for word in seg['words']:
            xword = word['word'][1:]
            amp_transcript['results']['words'].append({
                'type': "pronunciation",
                'text': xword,
                'start': word['start'],
                'end': word['end'],
                'offset': offset
            })                        
            tword = amp_transcript['results']['transcript'][offset:offset + len(xword)]
            if tword != xword:
                logging.warning(f"Transcript mismatch @{offset}: word='{xword}', transcript='{tword}'")
            else:
                logging.debug(f"Transcript correct: @{offset}: word={xword}, transcript={tword}")
            offset += len(word['word'])
            duration = max(duration, word['end'])
    
    amp_transcript['results']['duration'] = duration 
    amp_transcript['media']['duration'] = duration
    write_json_file(amp_transcript, amp_file)


def generate_amp_diarization(whisper_json, amp_diarization, input_media):
    "Generate an amp diarization file from the input"
    d = {
        'media': {'filename': input_media,
                  'duration': 0},
        'numSpeakers': 1,
        'segments': []
    }

    data = read_json_file(whisper_json)
    seg_start, seg_end = None, None
    for seg in data['segments']:
        if seg_start is None:
            # new diarization segment
            seg_start = seg['start']
            seg_end = seg['end']
        elif int(seg_end * 10) == int(seg['start'] * 10):
            # within 1/10th second we're the same, so continuation
            seg_end = seg['end']
        else:
            # there's a gap, so so write this one and start a new one
            d['segments'].append({'label': None,
                                  'start': seg_start,
                                  'end': seg_end,
                                  'speakerLabel': 'spk_0'})
            seg_start = seg['start']
            seg_end = seg['end']

    if seg_start is not None:
        d['segments'].append({'label': None,
                              'start': seg_start,
                              'end': seg['end'],
                              'speakerLabel': 'spk_0'})

    d['media']['duration'] = seg_end

    write_json_file(d, amp_diarization)


if __name__ == "__main__":
    main()
