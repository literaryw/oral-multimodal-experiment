"""
TCGA-HNSC 数据下载脚本
下载内容:
1. 临床数据 (TNM分期, 病理分级, 生存数据等)
2. 病理报告文本
3. WSI组织病理切片 (.svs文件)
"""
import json
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path("../data/TCGA-HNSC")
GDC_API = "https://api.gdc.cancer.gov"

def download_clinical_data():
    """下载TCGA-HNSC临床数据和病理报告"""
    print("=== 下载临床数据 ===")

    # GDC API: 获取HNSC项目的临床数据
    endpoint = f"{GDC_API}/cases"
    params = {
        "filters": json.dumps({
            "op": "in",
            "content": {
                "field": "project.project_id",
                "value": ["TCGA-HNSC"]
            }
        }),
        "fields": "submitter_id,diagnoses,exposures,demographic",
        "format": "JSON",
        "size": "1000"
    }

    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        data = response.json()
        print(f"  获取到 {len(data.get('data', {}).get('hits', []))} 条病例记录")

        # 解析诊断信息
        diagnoses = []
        for hit in data.get("data", {}).get("hits", []):
            case_id = hit.get("submitter_id")
            for diag in hit.get("diagnoses", []):
                diagnoses.append({
                    "case_id": case_id,
                    "primary_diagnosis": diag.get("primary_diagnosis"),
                    "tumor_stage": diag.get("tumor_stage"),
                    "tumor_grade": diag.get("tumor_grade"),
                    "ajcc_pathologic_t": diag.get("ajcc_pathologic_t"),
                    "ajcc_pathologic_n": diag.get("ajcc_pathologic_n"),
                    "ajcc_pathologic_m": diag.get("ajcc_pathologic_m"),
                    "ajcc_pathologic_stage": diag.get("ajcc_pathologic_stage"),
                })

        df = pd.DataFrame(diagnoses)
        df.to_csv(DATA_DIR / "clinical_data.csv", index=False)
        print(f"  临床数据已保存: {len(df)} 条诊断记录")
        return df
    else:
        print(f"  API请求失败: {response.status_code}")
        return None

def download_manifest():
    """获取HNSC项目的WSI文件清单"""
    print("\n=== 获取WSI文件清单 ===")

    endpoint = f"{GDC_API}/files"
    params = {
        "filters": json.dumps({
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-HNSC"]}},
                {"op": "in", "content": {"field": "files.data_type", "value": ["Slide Image"]}},
                {"op": "in", "content": {"field": "files.access", "value": ["open"]}}
            ]
        }),
        "fields": "file_id,file_name,cases.submitter_id,data_type,file_size",
        "format": "JSON",
        "size": "2000"
    }

    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        files = response.json().get("data", {}).get("hits", [])
        print(f"  找到 {len(files)} 个WSI文件")

        manifest = []
        for f in files:
            manifest.append({
                "file_id": f.get("file_id"),
                "file_name": f.get("file_name"),
                "case_id": f.get("cases", [{}])[0].get("submitter_id") if f.get("cases") else "",
                "file_size_mb": round(f.get("file_size", 0) / 1e6, 1)
            })

        df = pd.DataFrame(manifest)
        df.to_csv(DATA_DIR / "wsi_manifest.csv", index=False)

        total_gb = df["file_size_mb"].sum() / 1000
        print(f"  总大小: {total_gb:.1f} GB")
        print(f"  Manifest已保存: {len(df)} 文件")
        return df
    else:
        print(f"  API请求失败: {response.status_code}")
        return None

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: 下载临床数据
    clinical_df = download_clinical_data()

    # Step 2: 获取WSI清单
    manifest_df = download_manifest()

    print("\n=== 完成 ===")
    print(f"临床数据: {DATA_DIR / 'clinical_data.csv'}")
    print(f"WSI清单: {DATA_DIR / 'wsi_manifest.csv'}")
    print("\n注意: WSI文件较大(单个1-3GB)，建议选择性下载。")
    print("手动下载命令: ./gdc-client download -m gdc_manifest.txt")
