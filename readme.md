# Towards Fast, Accurate and Stable 3D Dense Face Alignment

[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
![GitHub repo size](https://img.shields.io/github/repo-size/babblelabs/3DDFA_V2.svg)

A fork of 3DDFA_V2, modified for internal purposes.

## Installation (Linux):
1. Clone the repository with `git clone`:

  `$ git clone --recursive git@github.com:babblelabs/3ddfa_v2`

  `$ cd 3ddfa_v2`

2. Create a virtual environment and activate it:

  `$ python3 -m venv 3ddfa-venv`

  `$ source 3ddfa-venv/bin/activate`

3. Install packages from the `requirements.txt` file into the environment:

`(venv) $ pip install -r requirements.txt`

4. Run the Cython build command:

  `$ sh ./build.sh`

## Usage:

### Video analysis:
Performs an analysis of a video file. Collects information about headcount, head position, orientation, and proximity.

Syntax:

`$ (venv) python3 demo_webcam_smooth.py [options]`

Currently working options are:

`-f`, `--video_fp`: path to the input video file.

`--dump_results`: can be `true` or `false`. Controls whether results are dumped to a file. If `true`, saves the results in comma-separated value format in `dumps/` directory.

`-c`, `--config`: config to use by 3DDFA network. Defines used checkpoint file, architecture, number of parameters and a few other variables. Defaults to `configs/mb1_120x120.yml`.

<!-- TODO: add description of the CSV file format -->

Upstream repository:
[3DDFA_v2]
by [Jianzhu Guo](https://guojianzhu.com), [Xiangyu Zhu](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/), [Yang Yang](http://www.cbsr.ia.ac.cn/users/yyang/main.htm), Fan Yang, [Zhen Lei](http://www.cbsr.ia.ac.cn/users/zlei/) and [Stan Z. Li](https://scholar.google.com/citations?user=Y-nyLGIAAAAJ).
The upstream code repo is owned and maintained by [Jianzhu Guo](https://guojianzhu.com).