# copyright idian group1
# Python implementation for '这b班上的值不值'

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="这b班上的值不值")

    parser.add_argument("--mean_daily", required=True,
                        help="平均日薪酬")
    parser.add_argument("--work_time", required=True,
                        help="工作时长")
    parser.add_argument("--traffic_time", required=True,
                        help="通勤时长")
    parser.add_argument("--fishing_time", required=True,
                        help="摸鱼时长")
    
    parser.add_argument("--degree_coef", required=True,
                        help="学历系数")
    parser.add_argument("--env_coef", required=True,
                        help="工作环境系数")
    parser.add_argument("--gender_coef", required=True,
                        help="学历系数")
    parser.add_argument("--co_coef", required=True,
                        help="学历系数")
    parser.add_argument("--qual_coef", required=True,
                        help="学历系数")
    parser.add_argument("--is_early", required=True,
                        help="是否8：30前上班")