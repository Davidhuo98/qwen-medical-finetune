"""
医疗数据集批量转换脚本
将 llm-medical-data 中的所有原始数据转换为 LLaMA-Factory 可用的 JSON 格式
脚本放到llm-medical-data/scripts目录下，然后运行
使用方法：
    cd llm-medical-data/scripts
    python convert_all.py

    # 或指定数据根目录
    python convert_all.py --data_dir /path/to/llm-medical-data-main
"""

import os
import subprocess
import sys
import argparse


def run(cmd, desc=""):
    """执行命令并打印结果"""
    print(f"\n{'─'*50}")
    if desc:
        print(f"▶ {desc}")
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, text=True)
    if result.returncode != 0:
        print(f"⚠️  命令执行失败（returncode={result.returncode}），跳过继续")
    else:
        print("✅ 完成")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="医疗数据集批量转换")
    parser.add_argument(
        "--data_dir",
        default="..",
        help="llm-medical-data-main 根目录路径（默认：上级目录）"
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    scripts_dir = os.path.dirname(os.path.abspath(__file__))

    csv_dialogue_dir = os.path.join(data_dir, "chinese_medical_dialogue_data")
    opengpt_dir = os.path.join(data_dir, "opengpt_data")

    csv_script = os.path.join(scripts_dir, "csv2json_chinese_medical_dialogue_data.py")
    opengpt_script = os.path.join(scripts_dir, "csv2json_opengpt_data.py")

    print("=" * 50)
    print("  医疗数据集批量转换")
    print("=" * 50)
    print(f"数据根目录: {data_dir}")
    print(f"脚本目录:   {scripts_dir}")

    # ── 1. 批量转换中文CSV ──────────────────────────────
    print("\n\n【第一步】转换 chinese_medical_dialogue_data CSV 文件")

    if not os.path.isdir(csv_dialogue_dir):
        print(f"⚠️  目录不存在，跳过: {csv_dialogue_dir}")
    else:
        csv_files = [f for f in os.listdir(csv_dialogue_dir) if f.endswith(".csv")]
        if not csv_files:
            print("⚠️  未找到CSV文件，跳过")
        else:
            for fname in sorted(csv_files):
                fpath = os.path.join(csv_dialogue_dir, fname)
                run(
                    f'python "{csv_script}" --rd_csv_path "{fpath}"',
                    desc=f"转换 {fname}"
                )

    # ── 2. 转换 opengpt 数据 ────────────────────────────
    print("\n\n【第二步】转换 opengpt_data 文件")

    opengpt_tasks = [
        (
            "tqa",
            os.path.join(opengpt_dir, "prepared_generated_data_for_nhs_uk_qa.csv"),
            "NHS UK QA 数据"
        ),
        (
            "ttask",
            os.path.join(opengpt_dir, "prepared_generated_data_for_medical_tasks.csv"),
            "Medical Tasks 数据"
        ),
        (
            "tchat",
            os.path.join(opengpt_dir, "prepared_generated_data_for_nhs_uk_conversations.csv"),
            "NHS UK 多轮对话数据（拆分为单轮）"
        ),
    ]

    if not os.path.isdir(opengpt_dir):
        print(f"⚠️  目录不存在，跳过: {opengpt_dir}")
    else:
        for subcmd, fpath, desc in opengpt_tasks:
            if not os.path.isfile(fpath):
                print(f"\n⚠️  文件不存在，跳过: {fpath}")
                continue
            run(
                f'python "{opengpt_script}" {subcmd} --rd_csv_path "{fpath}"',
                desc=desc
            )

    # ── 3. 汇总输出 ─────────────────────────────────────
    print("\n\n" + "=" * 50)
    print("  转换完成，生成的 JSON 文件：")
    print("=" * 50)

    for search_dir in [csv_dialogue_dir, opengpt_dir]:
        if not os.path.isdir(search_dir):
            continue
        for fname in sorted(os.listdir(search_dir)):
            if fname.endswith(".json"):
                fpath = os.path.join(search_dir, fname)
                size_mb = os.path.getsize(fpath) / 1024 / 1024
                print(f"  ✅ {fname}  ({size_mb:.1f} MB)")

    print("\n下一步：将所有 JSON 文件复制到 LLaMA-Factory/data/ 目录")
    print("命令示例：")
    print(f'  cp {csv_dialogue_dir}/*.json /path/to/LLaMA-Factory/data/')
    print(f'  cp {opengpt_dir}/*.json /path/to/LLaMA-Factory/data/')


if __name__ == "__main__":
    main()
