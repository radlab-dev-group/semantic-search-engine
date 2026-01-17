import json
import argparse
import pandas as pd
from aiohttp.client_reqrep import json_re


def prepare_parser(desc=""):
    p = argparse.ArgumentParser(description=desc)

    p.add_argument("--base-file", dest="base_file", required=True, type=str)
    p.add_argument(
        "--evaluated-file", dest="evaluated_file", required=True, type=str
    )
    p.add_argument(
        "--merged-out-file", dest="merged_out_file", required=True, type=str
    )
    return p


def get_avg_config_value(
    evaluated_examples: list[dict],
    main_key: list[str],
    exclude_collections: list[str],
):
    avg_cfg = {}
    avg_info = {}
    for e_example in evaluated_examples:
        if e_example["collection_name"] in exclude_collections:
            continue

        k_str = "#".join(str(e_example[m]) for m in main_key)
        if k_str not in avg_cfg:
            c_name_full = e_example["collection_name"]
            c_name_emb_version = c_name_full[: c_name_full.rfind("__")]
            chunk_size = c_name_full[c_name_full.rfind("__") + 2 :]
            c_name = c_name_emb_version[: c_name_emb_version.find("__")]
            emb_type = c_name_emb_version[c_name_emb_version.find("__") + 2 :]

            avg_info[k_str] = {
                "test_name": e_example["test_name"],
                "collection_name": c_name,
                "embedder_type": emb_type,
                "chunk_size": chunk_size,
                "sse_max_results": e_example["sse_max_results"],
                "gen_perc_rank_mass": e_example["gen_perc_rank_mass"],
            }

            avg_cfg[k_str] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

        avg_cfg[k_str][e_example["Ocena"]] += 1

    for k, v in avg_cfg.items():
        all_res_count = sum(v.values())
        sum_all_res = 0
        for eval_value, eval_count in v.items():
            sum_all_res += eval_count * eval_value
        avg_cfg[k]["avg"] = sum_all_res / all_res_count

        for i_k, i_v in avg_info[k].items():
            avg_cfg[k][i_k] = i_v

    return avg_cfg


def main(argv=None):
    args = prepare_parser(argv).parse_args(argv)

    e_dict = {}
    e_df = pd.read_excel(args.evaluated_file)
    for row in e_df.to_dict(orient="records"):
        e_dict[row["Idm"]] = row

    b_dict = {}
    b_df = pd.read_excel(args.base_file)
    for row in b_df.to_dict(orient="records"):
        b_dict[row["Idm"]] = row

    evaluated = []
    for idm_e, eval_res in e_dict.items():
        e_item = b_dict[idm_e]
        e_item["Ocena"] = eval_res["Ocena"]
        e_item["Komentarz"] = eval_res["Komentarz"]
        evaluated.append(e_item)

    # Store merged file
    pd.DataFrame(evaluated).to_excel(args.merged_out_file)

    main_key = [
        "collection_name",
        "test_name",
        "sse_max_results",
        "gen_perc_rank_mass",
    ]

    exclude_collection = [
        # "default_user_akozak20240918_e20240823__v2__denoiser__200_20"
    ]

    avg_config = get_avg_config_value(
        evaluated_examples=evaluated,
        main_key=main_key,
        exclude_collections=exclude_collection,
    )

    # Store resutls
    out_res_list = [v for v in avg_config.values()]
    summary_out_file_path = args.merged_out_file.replace(".xlsx", "-summary.xlsx")
    pd.DataFrame(out_res_list).to_excel(summary_out_file_path)


if __name__ == "__main__":
    main()
