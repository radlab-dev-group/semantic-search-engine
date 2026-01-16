import json
import logging
import Levenshtein
import pandas as pd
from torchmetrics.text import BLEUScore
from torchmetrics.text.rouge import ROUGEScore

from system.models import OrganisationUser

from engine.controllers.database import RelationalDBController

from engine.controllers.search import SearchQueryController
from engine.controllers.models_logic import GenerativeModelController
from engine.controllers.models_logic import EmbeddingModelsConfig

TEST_STEP_SEM_SEARCH = "semantic_search"
TEST_STEP_GEN_RESPONSE = "generative_response"

TEST_CONFIG_FIELD = "test_configuration"
SEM_SEARCH_RAG_TC = "semantic_search_rag"


class TestsLoader:
    TEST_CASES = [SEM_SEARCH_RAG_TC]

    def __init__(
        self,
        test_user: OrganisationUser,
        json_path: str,
        load_tests: bool = True,
        verify_collections: bool = True,
    ):
        self.json_path = json_path
        self.test_user = test_user
        self.verify_collections = verify_collections

        self._test_configuration = None
        self.test_cases_to_run = {n: {} for n in self.TEST_CASES}
        if load_tests:
            self.__load_tests_file(json_path=self.json_path)

        self.tc_ws = "  "
        self._embedding_config = EmbeddingModelsConfig()
        self._gen_model_controller = GenerativeModelController(store_to_db=True)

    def start(self, out_xlsx_file: str, out_xlsx_chunks_file: str):
        logging.info("#" * 100)
        logging.info(f">>> Starting tests from file {self.json_path} <<< ")

        rerank_results = [False]
        gen_models = self._test_configuration["config"]["generative_models"]
        max_results = self._test_configuration["config"]["search_options"][
            "max_results"
        ]
        percentage_rank_mass = self._test_configuration["config"]["search_options"][
            "percentage_rank_mass"
        ]

        all_results = []
        for rerank in rerank_results:
            for max_result in max_results:
                options_dict = {
                    "categories": [],
                    "documents": [],
                    "relative_paths": [],
                    "max_results": max_result,
                    "rerank_results": rerank,
                    "return_with_factored_fields": False,
                }
                for prm in percentage_rank_mass:
                    for gen_model in gen_models:
                        generative_options_dict = {
                            "generative_model": gen_model,
                            "percentage_rank_mass": prm,
                            "answer_language": "",
                            "translate_answer": False,
                            "use_doc_names_in_response": False,
                        }

                        logging.info("%%%%%%%%%% testing options %%%%%%%%%%")
                        logging.info(f"  - rerank               = {rerank}")
                        logging.info(f"  - max_result           = {max_result}")
                        logging.info(f"  - percentage_rank_mass = {prm}")
                        logging.info(f"  - generative model     = {gen_model}")
                        logging.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

                        for t_c_name, test_case in self.test_cases_to_run.items():
                            test_case_out = self._start_test_case(
                                name=t_c_name,
                                test_case=test_case,
                                options_dict=options_dict,
                                generative_options_dict=generative_options_dict,
                            )

                            all_results.append(
                                {
                                    "search_options_dict": options_dict,
                                    "generative_options_dict": generative_options_dict,
                                    "test_case_name": t_c_name,
                                    "test_case_out": test_case_out,
                                }
                            )

        self.__store_to_xlsx(
            out_xlsx_file=out_xlsx_file,
            out_xlsx_chunks_file=out_xlsx_chunks_file,
            all_results=all_results,
        )

        logging.info(f">>> Test from file {self.json_path} are passed <<< ")
        logging.info("#" * 100)

    def __store_to_xlsx(
        self, out_xlsx_file: str, out_xlsx_chunks_file: str, all_results: list
    ):
        all_res_dicts = []
        all_res_chunks_dicts = []
        for result in all_results:
            all_single_res, sse_chunks_res = self.__convert_result_to_list_dict(
                result
            )
            all_res_dicts.extend(all_single_res)
            all_res_chunks_dicts.extend(sse_chunks_res)

        uq_chunks_filename = out_xlsx_chunks_file.replace(".xlsx", "-unique.xlsx")
        answer_chunks_dicts = self.__merge_results_to_uniq_answer_chunks(
            results=all_res_chunks_dicts
        )

        pd.DataFrame(all_res_dicts).to_excel(out_xlsx_file, index=False)
        pd.DataFrame(all_res_chunks_dicts).to_excel(
            out_xlsx_chunks_file, index=False, engine="xlsxwriter"
        )
        pd.DataFrame(answer_chunks_dicts).to_excel(
            uq_chunks_filename, index=False, engine="xlsxwriter"
        )

    def __merge_results_to_uniq_answer_chunks(self, results: list[dict]) -> list:
        uq_results = []
        uq_res_dict = {}
        for res in results:
            question = res["question"]
            text_str = res["text_str"]
            left_context = res["left_context"]
            right_context = res["right_context"]
            test_name = res["test_name"]
            test_case_name = res["test_case_name"]
            document_name = res["document_name"]
            page_number = res["page_number"]
            text_number = res["text_number"]
            full_test_name = f"{test_case_name}_{test_name}"
            if full_test_name not in uq_res_dict:
                uq_res_dict[full_test_name] = []
            uq_res_dict[full_test_name].append(
                {
                    "question": question,
                    "document_name": document_name,
                    "text_str": text_str,
                    "right_context": right_context,
                    "left_context": left_context,
                    "page_number": page_number,
                    "text_number": text_number,
                }
            )

        # deduplicated results
        d_uq_res_dict = {}
        for f_t_name, all_texts in uq_res_dict.items():
            d_uq_res_dict[f_t_name] = self.__deduplicate_texts(texts=all_texts)

        for f_t_name, all_texts in d_uq_res_dict.items():
            for text_dict in all_texts:
                res_dict = {
                    "test_case_name": f_t_name,
                    "question": text_dict["question"],
                    "document_name": text_dict["document_name"],
                    "text_str": text_dict["text_str"],
                    "ocena_1__0": "",
                    "left_context": text_dict["left_context"],
                    "right_context": text_dict["right_context"],
                    "page_number": text_dict["page_number"],
                    "text_number": text_dict["text_number"],
                }
                uq_results.append(res_dict)
        return uq_results

    @staticmethod
    def __deduplicate_texts(texts: list[dict]) -> list[dict]:
        d_texts = []
        d_texts_str = []
        for t_dict in texts:
            # Skip if same text exists
            if t_dict["text_str"] in d_texts_str:
                continue
            d_texts.append(t_dict)
            d_texts_str.append(t_dict["text_str"])

        # Deduplicate d_texts with embedder?
        # d_texts = self._deduplicate_texts_embedder(texts=d_texts)

        return d_texts

    def __convert_result_to_list_dict(self, result: dict):
        test_case_name = result["test_case_name"]
        sse_max_results = result["search_options_dict"]["max_results"]
        sse_rerank_results = result["search_options_dict"]["rerank_results"]

        generative_model = result["generative_options_dict"]["generative_model"]
        percentage_rank_mass = result["generative_options_dict"][
            "percentage_rank_mass"
        ]

        all_single_res = []
        all_sse_found_chunks = []
        for example in result["test_case_out"]:
            for collection_name, c_results in example.items():
                for test_name, test_results in c_results.items():
                    for single_test_answer in test_results:
                        gen_answer_number = 1

                        sse_detailed_results = single_test_answer["semantic_search"][
                            "sse_detailed_results"
                        ]
                        user_query = single_test_answer["semantic_search"][
                            "user_query"
                        ]
                        perc_found = single_test_answer["semantic_search"][
                            "human_answers"
                        ]["1"]["perc_found"]

                        single_res = {
                            "id": "<id>",
                            "query_response_id": "<response id>",
                            "test_case_name": test_case_name,
                            "test_name": test_name,
                            "sse_max_results": sse_max_results,
                            "sse_rerank_results": sse_rerank_results,
                            "gen_perc_rank_mass": percentage_rank_mass,
                            "collection_name": collection_name,
                            "gen_generative_model": generative_model,
                            "question": user_query,
                            "human_answer": "<no human answer>",
                            "generated_answer_number": gen_answer_number,
                            "generated_answer": "<not generated>",
                            "perc_found": perc_found,
                        }

                        if sse_detailed_results is not None:
                            sse_chunks = self._convert_to_sse_chunks_found(
                                single_res=single_res.copy(),
                                sse_detailed_results=sse_detailed_results,
                            )
                            if len(sse_chunks):
                                all_sse_found_chunks.extend(sse_chunks)

                        if "generative_response" not in single_test_answer:
                            all_single_res.append(single_res.copy())
                            continue

                        for response in single_test_answer["generative_response"]:
                            query_response_id = response["query_response_id"]
                            question = response["question"]
                            human_answer = response["human_answer"]
                            generated_answer = response["generated_answer"]
                            single_res["id"] = response["id"]
                            single_res["query_response_id"] = query_response_id
                            single_res["question"] = question
                            single_res["human_answer"] = human_answer
                            single_res["generated_answer"] = generated_answer
                            single_res["generated_answer_number"] = gen_answer_number
                            for k in response.keys():
                                if k in [
                                    "question",
                                    "human_answer",
                                    "generated_answer",
                                ]:
                                    continue
                                single_res[k] = response[k]
                            all_single_res.append(single_res.copy())
                            gen_answer_number += 1
        return all_single_res, all_sse_found_chunks

    @staticmethod
    def _convert_to_sse_chunks_found(single_res: dict, sse_detailed_results: dict):
        """
        single_res = {
            "test_case_name": test_case_name,
            "test_name": test_name,
            "sse_max_results": sse_max_results,
            "sse_rerank_results": sse_rerank_results,
            "gen_perc_rank_mass": percentage_rank_mass,
            "collection_name": collection_name,
            "gen_generative_model": generative_model,
            "question": "",
            "human_answer": "",
            "generated_answer_number": gen_answer_number,
            "generated_answer": "<not generated>",
            "perc_found": perc_found,
        }
        :param single_res:
        :param sse_detailed_results:
        :return:
        """

        for k in [
            "id",
            "query_response_id",
            "gen_perc_rank_mass",
            "gen_generative_model",
            "human_answer",
            "generated_answer_number",
            "generated_answer",
            "perc_found",
        ]:
            if k in single_res:
                single_res.pop(k)

        all_results = []
        for result in sse_detailed_results.values():
            res_dict = single_res.copy()
            res_dict["document_name"] = result["document_name"]
            res_dict["page_number"] = result["page_number"]
            res_dict["text_number"] = result["text_number"]
            res_dict["text_str"] = result["text_str"]
            res_dict["left_context"] = result["left_context"]
            res_dict["right_context"] = result["right_context"]
            all_results.append(res_dict)
        return all_results

    def _start_test_case(
        self,
        name,
        test_case: dict,
        options_dict: dict,
        generative_options_dict: dict,
    ):
        test_case_out = []
        logging.info(f"{self.tc_ws} -> Starting test case {name}")
        if name == SEM_SEARCH_RAG_TC:
            ssr_tc = self._start_sem_rag_tc(
                test_case_dict=test_case,
                options_dict=options_dict,
                generative_options_dict=generative_options_dict,
            )
            test_case_out.append(ssr_tc)
        return test_case_out

    def _start_sem_rag_tc(
        self, test_case_dict: dict, options_dict: dict, generative_options_dict: dict
    ):
        rag_tc_out = {}
        collections = test_case_dict["collections"]
        test_examples = test_case_dict["test_examples"]

        run_examples = test_case_dict["run_examples"]
        if not len(run_examples):
            run_examples = list(test_examples.keys())

        logging.info(f"{2 * self.tc_ws} - number of collections: {len(collections)}")
        logging.info(
            f"{2 * self.tc_ws} - number of test examples: {len(test_examples)}"
        )

        for c_name in collections:
            rag_tc_out[c_name] = {}
            collection = RelationalDBController.get_collection(
                created_by=self.test_user, collection_name=c_name
            )
            if collection is None:
                logging.error(f"Collection {c_name} not found!")
                raise Exception(f"Collection {c_name} not found!")

            logging.info(f"{2 * self.tc_ws}" + "-" * 50)
            logging.info(f"{2 * self.tc_ws} Testing collection: {c_name}")
            for e_name in run_examples:
                if e_name not in test_examples:
                    raise Exception(f"Test example {e_name} not found!")

                example = test_examples[e_name]
                logging.info(f"{2 * self.tc_ws}\texample: {e_name}")
                rag_tc_e_res = self._do_start_sem_rag_tc_example(
                    example=example,
                    collection=collection,
                    options_dict=options_dict,
                    generative_options_dict=generative_options_dict,
                )
                rag_tc_out[c_name][e_name] = rag_tc_e_res
        return rag_tc_out

    def _do_start_sem_rag_tc_example(
        self,
        example: dict,
        collection,
        options_dict: dict,
        generative_options_dict: dict,
    ):
        questions = example["questions"]
        human_answers = example["human_answers"]
        logging.info(f"{2 * self.tc_ws}\t * questions: {len(questions)}")
        logging.info(f"{2 * self.tc_ws}\t * answers: {len(human_answers)}")

        out_rag_responses = []
        for question in questions:
            q_res = self._do_start_sem_rag_tc_example_question(
                question=question,
                human_answers=human_answers,
                options_dict=options_dict,
                collection=collection,
                generative_options_dict=generative_options_dict,
            )
            out_rag_responses.append(q_res)
        return out_rag_responses

    def _do_start_sem_rag_tc_example_question(
        self,
        question: str,
        human_answers: dict,
        options_dict: dict,
        collection,
        generative_options_dict: dict,
    ):
        logging.info(f"{3 * self.tc_ws}\t -> question: {question}")
        out_q_res = {}

        if TEST_STEP_SEM_SEARCH not in self._test_configuration["what_to_test"]:
            logging.warning(f"{TEST_STEP_SEM_SEARCH} is not set, test is skipped!")
            return

        results = SearchQueryController.new_query(
            query_str=question,
            search_options_dict=options_dict,
            collection=collection,
            organisation_user=self.test_user,
            sse_engin_config_path="configs/milvus_config.json",
            ignore_question_lang_detect=True,
        )
        if "results" not in results:
            raise Exception("results field not found in response!")

        # ##### SEARCH TEST
        out_q_res[TEST_STEP_SEM_SEARCH] = {}
        m_results = results["results"]
        general_stats = m_results["stats"]
        detailed_results = m_results["detailed_results"]

        search_files = [f for f in list(general_stats.keys())]
        out_q_res[TEST_STEP_SEM_SEARCH]["search_files"] = search_files
        ff_str = "\n[##] ".join(sorted(search_files))

        out_q_res[TEST_STEP_SEM_SEARCH]["human_answers"] = {}
        out_q_res[TEST_STEP_SEM_SEARCH]["user_query"] = m_results["query"]
        out_q_res[TEST_STEP_SEM_SEARCH]["sse_detailed_results"] = detailed_results
        for ans_id, ans in human_answers.items():
            out_q_res[TEST_STEP_SEM_SEARCH]["human_answers"][ans_id] = {}
            human_files = []
            found_files = []
            not_found_files = []
            for file_conf in ans["files"]:
                file_name = file_conf["file_name"]
                exact_match_file_name = file_conf["exact_match_file_name"]
                if exact_match_file_name:
                    if file_name in search_files:
                        found_files.append(file_name)
                    else:
                        not_found_files.append(file_name)
                else:
                    found_l_file = False
                    for s_f in search_files:
                        lev_dist = Levenshtein.distance(s_f, file_name)
                        lev_val = float(lev_dist) / len(file_name)
                        if lev_val < 0.1:
                            found_files.append(file_name)
                            found_l_file = True
                            break
                    if not found_l_file:
                        not_found_files.append(file_name)
                human_files.append(file_name)
            out_q_res[TEST_STEP_SEM_SEARCH]["human_answers"][ans_id] = {
                "human_files": human_files,
                "found_files": found_files,
                "not_found_files": not_found_files,
                "perc_found": float(len(found_files)) / len(human_files),
            }

            hf_str = ""
            for f_name in human_files:
                pref_status = "[  ]"
                if f_name in found_files:
                    pref_status = "[OK]"
                hf_str += f"{pref_status} {f_name}\n"
            logging.info(f"{3 * self.tc_ws}\t -> human files:\n{hf_str}")
            logging.info(f"{3 * self.tc_ws}\t -> search files:\n{ff_str}")
            logging.info(f"{3 * self.tc_ws}\t {50 * '-'}")

        # ##### GENERATIVE RESPONSE TEST
        if TEST_STEP_GEN_RESPONSE in self._test_configuration["what_to_test"]:
            gen_evaluators = self._test_configuration["evaluators"]["generative"]
            query_response_id = results["query_response_id"]
            user_response = SearchQueryController.get_user_response_by_id(
                query_response_id=query_response_id
            )

            query_responses = []
            for i in range(
                self._test_configuration["config"]["generate_options"][
                    "answers_count"
                ]
            ):
                query_response = (
                    self._gen_model_controller.generative_answer_for_response(
                        user_response=user_response,
                        query_instruction=question,
                        query_options=generative_options_dict,
                    )
                )
                if query_response is None:
                    logging.warning(f"Received None query response!")
                    continue

                logging.info(
                    f"{3 * self.tc_ws}\t -> generated answer:\n"
                    f"{query_response.generated_answer}"
                )
                query_responses.append(query_response)

            gen_resp_res = []
            for query_response in query_responses:
                for ans in human_answers.values():
                    logging.info(
                        f"{3 * self.tc_ws}\t -> human answer:\n"
                        f"{ans['text_answer']}"
                    )

                    single_res = {
                        "id": len(gen_resp_res),
                        "query_response_id": query_response_id,
                        "question": question,
                        "generated_answer": query_response.generated_answer,
                        "human_answer": ans["text_answer"],
                    }
                    for g_metric in gen_evaluators:
                        if g_metric == "rouge":
                            rouge = ROUGEScore()
                            rouge_score = rouge(
                                query_response.generated_answer, ans["text_answer"]
                            )
                            logging.info(f"{3 * self.tc_ws}\t -> ROUGE_METRIC:")
                            for r_name, r_val in rouge_score.items():
                                logging.info(
                                    f"{4 * self.tc_ws}\t\t -> {r_name}: {r_val.item()}"
                                )
                                single_res[r_name] = r_val.item()
                        elif g_metric == "bleu":
                            for i in range(1, 5):
                                bleu = BLEUScore(n_gram=i)
                                r_name = f"{g_metric}_{i}"
                                bleu_score = bleu(
                                    query_response.generated_answer,
                                    [ans["text_answer"]],
                                )
                                single_res[r_name] = bleu_score.item()
                                logging.info(
                                    f"{4 * self.tc_ws}\t\t -> "
                                    f"{r_name}: {bleu_score.item()}"
                                )
                        elif g_metric == "cross-encoder":
                            pass
                    gen_resp_res.append(single_res)
            out_q_res[TEST_STEP_GEN_RESPONSE] = gen_resp_res
        return out_q_res

    def __load_tests_file(self, json_path: str | None):
        if json_path is not None and len(json_path):
            self.json_path = json_path
        assert self.json_path is not None and len(self.json_path)
        with open(self.json_path, "r") as f:
            whole_config_json = json.load(f)

        self.__load_test_configuration(whole_config_json)

        for test_case in self.TEST_CASES:
            if test_case not in whole_config_json:
                continue
            self.test_cases_to_run[test_case] = whole_config_json[test_case]
            if self.verify_collections:
                self.__check_collection(
                    self.test_cases_to_run[test_case]["collections"]
                )

    def __load_test_configuration(self, whole_config_json: dict):
        self._test_configuration = whole_config_json[TEST_CONFIG_FIELD]

    def __check_collection(self, collections: list[str]):
        found_collections = []
        not_found_collections = []
        for c_name in collections:
            collection = RelationalDBController.get_collection(
                created_by=self.test_user, collection_name=c_name
            )
            if not collection:
                not_found_collections.append(c_name)
            else:
                found_collections.append(c_name)

        logging.info("Found collections:")
        for c in found_collections:
            logging.info(f"\t[*] {c}")

        logging.info("Not found collections:")
        for c in not_found_collections:
            logging.info(f"\t[!] {c}")

        if len(not_found_collections):
            raise SystemExit(
                f"Exists {len(not_found_collections)} not found collections"
            )
