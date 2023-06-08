#!/usr/bin/env python3
# Driver for whisper.sif
import os.path
import shutil
import subprocess
import sys
import tempfile
import argparse
import logging
from pathlib import Path
import math
import atexit
import os
import time




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true", help="Turn on debugging")
    parser.add_argument("input_audio", help="Input audio file")
    parser.add_argument("kaldi_transcript_json", help="Output Kaldi Transcript JSON file")
    parser.add_argument("kaldi_transcript_text", help="Output Kaldi Transcript Text file")
    parser.add_argument("--overlay_dir", default=None, nargs=1, help="Directory for the overlay file (default to cwd)")
    args = parser.parse_args()    
    logging.basicConfig(format="%(asctime)s [%(levelname)-8s] (%(filename)s:%(lineno)d:%(process)d)  %(message)s", level=logging.DEBUG if args.debug else logging.INFO)   
    logging.info(f"Starting with args {args}")

    # copy the input file to a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:        
        shutil.copy(args.input_audio, f"{tmpdir}/xxx.wav")
        # find the right Kaldi SIF and set up things to make it work...
        sif = Path(sys.path[0], f"kaldi-pua-{'gpu' if args.gpu else 'cpu'}.sif")
        if not sif.exists():
            logging.error(f"Kaldi SIF file {sif!s} doesn't exist!")
            exit(1)

        # By default, singularity will map $HOME, /var/tmp and /tmp to somewhere outside
        # the container.  That's good.  
        # The authors of the kaldi docker image assumed that they could write anywhere
        # they pleased on the container image.  That's bad.
        # With the --writable-tmpfs, singularity will produce a 16M overlay filesystem that
        # handles writes everywhere else.  That's good.
        # BUT kaldi writes big files all over the place...and they will routinely exceed
        # 16M.  That's bad.  
        # The bottom line is that we have to create an overlay image that's big enough
        # for what we're trying to do.  But how big?  No matter what size we use, it will
        # never be enough for some cases.  For now, let's look at the size of the input file
        # (which should be a high-bitrate wav) and use some multiple.  Empirically, it looks
        # like 10x should do the trick. 
        overlay_size = math.ceil((10 * Path(args.input_audio).stat().st_size) / 1048576)
        if overlay_size < 64:
            overlay_size = 64
        if args.overlay_dir is None:
            args.overlay_dir = str(Path('.').absolute())
        else:
            args.overlay_dir = str(Path(args.overlay_dir[0]).absolute())
        overlay_file = f"{args.overlay_dir}/kaldi-overlay-{os.getpid()}-{time.time()}.img"
        
        if not args.debug:
            # make sure to erase the overlay at the end.  This is kind of an abuse of lambda...
            atexit.register(lambda: Path(overlay_file).unlink() if Path(overlay_file).exists() else None)
        try:
            subprocess.run(["singularity", "overlay", "create", "--size", str(overlay_size), overlay_file], check=True)
            logging.debug(f"Created overlay file {overlay_file} {overlay_size}MB")
        except subprocess.CalledProcessError as e:
            logging.exception(f"Cannot create the overlay image of {overlay_size} bytes as {overlay_file}!")            
            exit(1)

        # build the singularity command line
        cmd = ['singularity', 'run', '-B', f"{tmpdir}:/audio_in", '--overlay', overlay_file, str(sif) ]
        logging.debug(f"Singularity Command: {cmd}")
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')        
        if p.returncode != 0:
            logging.error("KALDI failed")
            logging.error(p.stdout)
            exit(1)
        copy_failed = False
        for src, dst in ((f"{tmpdir}/transcripts/txt/xxx_16kHz.txt", args.kaldi_transcript_text),
                         (f"{tmpdir}/transcripts/json/xxx_16kHz.json", args.kaldi_transcript_json)):
            try:                
                shutil.copy(src, dst)
            except Exception as e:
                logging.exception(f'Cannot copy {src} to {dst}')     
                copy_failed = True
        if copy_failed:
            logging.error("Kaldi didn't actually produce the files on a 'successful' run")
            logging.error(f"Kaldi's output:\n{p.stdout}")

    logging.debug(f"Make sure to manually remove the overlay file: {overlay_file}")
    logging.info(f"Finished running Kaldi with output {args.kaldi_transcript_json} and {args.kaldi_transcript_text}")
    exit(0 if not copy_failed else 1)

if __name__ == "__main__":
    main()
