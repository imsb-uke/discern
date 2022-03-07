#!/usr/bin/env python3
"""Merge json parameter files."""
import argparse
import json
import os.path
import pathlib


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Merge json parameter files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("file", nargs="+", help="Filenames to merge")
    filenames = parser.parse_args().file
    data = [json.loads(pathlib.Path(file).read_bytes()) for file in filenames]
    paths = [pathlib.Path(d["exp_param"]["experiments_dir"]) for d in data]
    commonpath = pathlib.Path(*os.path.commonprefix([d.parts for d in paths]))
    result = {
        "exp_param": {
            "experiments_dir": str(commonpath)
        },
        "experiments": {}
    }
    for file in data:
        rootpath = pathlib.Path(file["exp_param"]["experiments_dir"])
        for experimentname, experiment in file["experiments"].items():
            path = str(
                rootpath.joinpath(experimentname).relative_to(commonpath))
            if path in result["experiments"]:
                raise RuntimeError(f"Experiment {path} specified twice")
            result["experiments"][path] = experiment
    print(json.dumps(result))


if __name__ == "__main__":
    main()
