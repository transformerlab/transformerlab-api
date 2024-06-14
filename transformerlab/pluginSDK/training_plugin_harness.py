"""
This is the main file that gets called by popen when a training plugin is run.
It will get passed:

    --plugin_dir            full path to the directory containing the plugin
    --input_file            full path to the input configuration for the plugin
    --experiment_name       string name of experiment

The plugin expects that input_file and experiment_file are passed on.

"""
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--plugin_dir', type=str)
args, unknown = parser.parse_known_args()

# Add the plugin directory to the path
# Note that this will allow the plugin to import files in this file's directory
# So the plugin is able to import the SDK
sys.path.append(args.plugin_dir)
import main
