#!/bin/env python3
#
# Build the amp_mgms-whisper package
#
import argparse
import logging
import tempfile
from pathlib import Path
import os
import shutil
import sys
import subprocess
from amp.package import *

VERSION="1.0.0"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False, action='store_true', help="Turn on debugging")
    parser.add_argument('--package', default=False, action='store_true', help="build a package instead of installing")
    parser.add_argument('destdir', help="Output directory for the package or install")
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s [%(levelname)-8s] (%(filename)s:%(lineno)d)  %(message)s",
                        level=logging.DEBUG if args.debug else logging.INFO)

    # Building whisper is actually generating the .sif for whisper and copying
    # a few files around.
    sif_file = Path(sys.path[0], "whisper.sif")
    recipe_file = Path(sys.path[0], "whisper.recipe")

    if not sif_file.exists() or sif_file.stat().st_mtime < recipe_file.stat().st_mtime:
        logging.info("(Re)Building the .sif because it doesn't exist or is out of date")
        try:
            # This is a really big .sif -- let's make sure the TMPDIR is set
            # to this directory if it isn't set elsewhere
            if 'APPTAINER_TMPDIR' not in os.environ:
                os.environ['APPTAINER_TMPDIR'] = tempfile.TemporaryDirectory(dir=sys.path[0], prefix="apptainer-tmp")
                Path(os.environ['APPTAINER_TMPDIR']).mkdir(exist_ok=True)
                logging.info(f"Setting APPTAINER_TMPDIR = {os.environ['APPTAINER_TMPDIR']}")

            subprocess.run(['apptainer', 'build', '--force', str(sif_file), str(recipe_file)],
                           check=True)
        except Exception as e:
            logging.error(f"Failed to build apptainer: {e}")
            exit(1)

    try:
        build_dest = Path(tempfile.TemporaryDirectory(prefix="whisper-build-").name if args.package else args.destdir)
        (build_dest / "tools/whisper").mkdir(exist_ok=True, parents=True)
        
        logging.info(f"Copying files to {build_dest}")
        
        for f in ('whisper.sif', 'whisper.xml', 'whisper.py'):
            shutil.copy(Path(sys.path[0], f), build_dest / "tools/whisper" / f)
    except Exception as e:
        logging.error(f"Failed to copy files to {build_dest}: {e}")
        exit(1)

    if args.package:
        try:
            new_package = create_package("amp_mgms-whisper", VERSION, "galaxy",
                                         Path(args.destdir), build_dest,
                                         arch_specific=True, depends_on=['galaxy', 'amp_python']) 
            logging.info(f"New package in {new_package}")    
        except Exception as e:
            logging.error(f"Failed to build backage: {e}")
            exit(1)
    



if __name__ == "__main__":
    main()
