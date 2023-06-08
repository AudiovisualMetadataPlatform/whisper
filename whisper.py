#!/usr/bin/env python3
# Driver for whisper.sif
import shutil
import subprocess
import sys
import tempfile
import argparse
import logging
from pathlib import Path
import json


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
    parser.add_argument("--amp_transcript", type=str, help="Output AMP Transcript")
    parser.add_argument("--webvtt", type=str, help="WebVTT output")    
    parser.add_argument("--language", choices=whisper_languages, default="Auto", help="Audio Language")
    parser.add_argument("--model", choices=whisper_models, default='small', help="Language model to use")
    args = parser.parse_args()    
    logging.basicConfig(format="%(asctime)s [%(levelname)-8s] (%(filename)s:%(lineno)d:%(process)d)  %(message)s", level=logging.DEBUG if args.debug else logging.INFO)   
    logging.info(f"Starting with args {args}")

    if args.transcript_json is None and args.transcript_text is None and args.amp_transcript is None and args.webvtt is None:
        logging.error("You must select an output of some sort!")
        exit(1)

    sif = Path(sys.path[0], "whisper.sif")
    if not sif.exists():
        logging.error(f"Whisper SIF file {sif!s} not found")
        exit(1)

    with tempfile.TemporaryDirectory(prefix="whisper-") as tmpdir:
        logging.debug(f"Temporary directory: {tmpdir}")
        whisper_args = [#"--model_dir", whisper_model_dir,
                        "--output_dir", tmpdir,
                        "--model", args.model,
                        "--word_timestamps", "True",
                        args.input_media]
        if args.language != "Auto":
            whisper_args.extend(['--language', args.language])
        try:
            subprocess.run([str(sif), *whisper_args], check=True)
        except Exception as e:
            logging.exception(f"Failed to transcribe: {e}")
            exit(1)
                        
        try:
            # go through each of the requested outputs and gather them.
            if args.transcript_json:
                shutil.copy(get_file_by_ext(tmpdir, 'json'), args.transcript_json)

            if args.transcript_text:
                shutil.copy(get_file_by_ext(tmpdir, 'txt'), args.transcript_text)

            if args.webvtt:
                shutil.copy(get_file_by_ext(tmpdir, 'vtt'), args.webvtt)

            if args.amp_transcript:
                # convert the native json transcript to an amp transcript
                with open(get_file_by_ext(tmpdir, 'json')) as f:
                    data = json.load(f)
                amp_transcript = {
                    'media': {
                        'filename': args.input_media,
                        'duration': 0
                    },
                    'results': {
                        'transcript': data['text'],
                        'words': []
                    }
                }                
                duration = 0
                for seg in data['segments']:
                    for word in seg['words']:
                        amp_transcript['results']['words'].append({
                            'type': "",
                            'text': word['word'],
                            'start': word['start'],
                            'end': word['end']
                        })
                        duration = max(duration, word['end'])
                
                amp_transcript['results']['duration'] = duration 

                with open(args.amp_transcript, "w") as f:
                    json.dump(amp_transcript, f)

        except Exception as e:
            logging.exception(f"Failed to gather outputs: {e}")
            exit(1)


def get_file_by_ext(path, ext):
    files = list(Path(path).glob(f"*.{ext}"))
    if files:
        return files[0]
    else:
        raise FileNotFoundError(f"Cannot find file with extension {ext} in {path}")

if __name__ == "__main__":
    main()
